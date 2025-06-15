import os
import time
from functools import partial

import click
import facer
import numpy as np
import gradio as gr
import torch
from pathlib import Path

import torchvision
from PIL import Image
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from insightface.app import FaceAnalysis
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from face_swap import face_swap
from nearest_face import AnnoyWrapper
from pipelines import StableDiffusionImg2ImgPipelineRes
from utils import IPAdapterPlusFT


def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def write_crop(image, crop_path, face, embed_path):
    face = {key: value for key, value in face.items() if key != "embedding"}
    np.save(embed_path, face)
    Image.fromarray(image).crop(face["bbox"]).save(crop_path)


def runCAMOUFLAGE(
        image,
        max_size,
        threshold,
        num_inference_steps,
        alpha,
        strength,
        guidance_scale,
        seed,
        app,
        t,
        device,
        image_list,
        ip_model,
        face_attr,
):
    start_time = time.time()
    tfms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(max_size),
        torchvision.transforms.CenterCrop((max_size, max_size)),
    ])

    image = tfms(image)

    faces = app.get(np.array(image))
    faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    new_image = np.array(image.copy())
    face_used = []
    # blur_image = image.copy()
    if len(faces) == 0:
        res = None
    else:
        for i, face in enumerate(faces):
            rect = face.bbox.tolist()
            if (rect[2] - rect[0]) * (rect[3] - rect[1]) < 64 * 64:
                continue
            face_im = image.crop(rect)
            new_face = t.get_nns_by_vector(face_im, n=20, search_k=-1, include_distances=True)
            threshold = min(max(new_face[0]), threshold)
            new_face = list(zip(*new_face))
            new_faceb = [x for x in new_face if x[1] > threshold]
            new_face.reverse()
            new_face = (new_faceb if len(new_faceb) != 0 else new_face)[0][0]

            path = image_list[new_face]
            new_face = AttributeDict(
                np.load(path.parent.with_name(f"{path.parent.name}_embeddings") / f"{path.stem}.npy",
                        allow_pickle=True).item())
            fim = np.array(Image.open(path).convert("RGB"))
            face_used.append(path)

            new_image = face_swap(new_image, face, fim, new_face, True)

            # shape = (ceil(rect[2] - rect[0]), ceil(rect[3] - rect[1]))
            # tmp = color_transfer(np.array(face),
            #                      np.array(Image.open(image_list[new_face[0]]).convert("RGB").resize(shape)), True,
            #                      False)
            # new_face.embedding += torch.rand(new_face.embedding.shape).numpy() * new_face.embedding / max(abs(new_face.embedding.max()),abs(abs(new_face.embedding.max())))
            # face.embedding = torch.lerp(torch.tensor(face.embedding), torch.tensor(new_face.embedding), 0.8).numpy()
            # new_image = swapper.get(new_image, face, new_face, paste_back=True)
            # new_image.paste(tmp, (ceil(rect[0]), ceil(rect[1])), mask.resize(shape))
            # blur_image, new_image = swap_faces(new_image, fim, face, new_face, blur_image, edit=True)
            # blur_image, new_image = swap_face(new_image, face, blur_image)
        new_image = Image.fromarray(new_image)

        if ip_model.t2i_adapter is None:
            res = None
        else:
            raw_image = to_tensor(new_image).unsqueeze(0).to(device) * 255
            h, w = raw_image.shape[-2:]
            latent_shape_h, latent_shape_w = h // 8, w // 8
            latent_max = max(latent_shape_h, latent_shape_w)
            shape_max = max(h, w)
            res = torch.zeros((1, 41, latent_shape_h, latent_shape_w))
            ids = [0 for _ in range(len(faces))]
            rects = torch.stack([torch.tensor(f["bbox"]) for f in faces])
            points = torch.stack([torch.tensor(f["kps"]) for f in faces])
            faces = face_attr(raw_image, {'rects': rects, "points": points, "image_ids": ids})
            attrs = faces["attrs"]
            # torch.save(attrs, out / "attrs.pt")
            for n, i in enumerate(ids):
                p = torch.nn.functional.hardtanh(points[n] * (latent_max - 1) // shape_max, 0,
                                                 latent_max - 1).to(int)
                r = torch.nn.functional.hardtanh(rects[n] * latent_max // shape_max, 0, latent_max).to(int)
                res[i, -1, p[:, 1], p[:, 0]] = 1
                res[i, :40, max(r[1], 0):min(r[3], latent_shape_h), max(r[0], 0):min(r[2], latent_shape_w)] = \
                    attrs[
                        n].repeat(
                        min(r[3], latent_shape_h) - max(r[1], 0), min(r[2], latent_shape_w) - max(r[0], 0),
                        1).permute(
                        2, 0, 1)

    if isinstance(new_image, np.ndarray):
        new_image = Image.fromarray(new_image)
    images = ip_model.generate(pil_image=new_image, num_samples=1, num_inference_steps=num_inference_steps,
                               seed=seed,
                               guidance_scale=guidance_scale,
                               prompt="", negative_prompt="easyneg", image=Image.blend(new_image, image, alpha),
                               strength=strength, scale=1,
                               down_block_additional_residuals=None if res is None else ip_model.t2i_adapter(
                                   res.to(device, torch.float16)))

    return images[0], f"Time: {time.time() - start_time:.2f} s"


@click.command()
@click.option('--crop_path', type=click.Path(), help='Path to folder with crops. Images for face swap.',
              default='./FALCO/fake512')
@click.option('--method', type=str, help='Method to use for embedding',
              default="CLIP:Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64")
@click.option('--file_index', type=click.Path(exists=False), help='Path to file index',
              default="../fake_farl.ann")
@click.option('--batch_size', type=int, help='Batch size', default=128)
@click.option('--base_model_path', type=str, help='Path to base model',
              default="stablediffusionapi/realistic-vision-v51")
@click.option('--vae_model_path', type=str, help='Path to VAE model', default="stabilityai/sd-vae-ft-mse")
@click.option('--image_encoder_path', type=str, help='Path to image encoder',
              default="Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64")
@click.option('--ip_ckpt', type=str, help='Path to IP checkpoint', default="./light_1.5v.bin")
@click.option('--device', type=str, help='Device', default="cuda" if torch.cuda.is_available() else "cpu")
@click.option('--negative_path', type=click.Path(exists=True), help='Path to negative images',
              default="E:\projects\StellaProject\stable-diffusion-webui\embeddings\EasyNegative.safetensors")
def main(crop_path, method, file_index, batch_size, base_model_path, vae_model_path, image_encoder_path, ip_ckpt,
         device, negative_path):
    original_path = Path(crop_path)
    ip_ckpt = Path(ip_ckpt)
    negative_path = Path(negative_path)
    file_index = Path(file_index)
    assert original_path.exists() and ip_ckpt.exists() and negative_path.exists(), "Paths not found"

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(512, 512))

    crop_path = original_path.with_name(f"{original_path.name}_crop")
    embeddings_path = original_path.with_name(f"{original_path.name}_embeddings")
    crop_path.mkdir(parents=True, exist_ok=True)
    embeddings_path.mkdir(parents=True, exist_ok=True)

    image_list = [p for p in original_path.glob('*.*') if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    assert len(image_list) > 0, "No images found"

    for path in tqdm(image_list, desc="Checking cropped images..."):
        p_crop = crop_path / f"{path.stem}.png"
        p_emb = embeddings_path / f"{path.stem}.npy"
        if p_crop.exists() and p_emb.exists():
            continue
        image = np.array(Image.open(path).convert("RGB"))
        face = app.get(image, 1)
        if len(face) == 0:
            tqdm.write(f"Face not found in {path}")
            path.unlink()
        else:
            write_crop(image, p_crop, face[0], p_emb)

    t = AnnoyWrapper(method, device)

    if file_index.exists():
        t.annoy.load(str(file_index))
        print("Index loaded")
    else:
        print("Index not found! Generating...")
        t.generate(crop_path, batch_size=batch_size, name=str(file_index))
        print("Index generated")

    # if not windows
    run_compile = True if os.name != 'nt' else False

    # load VAE
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    pipe = StableDiffusionImg2ImgPipelineRes.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.to(memory_format=torch.channels_last)
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++",
                                                        use_karras_sigmas='true')
    pipe.scheduler = scheduler

    if negative_path is not None and negative_path.exists():
        pipe.load_textual_inversion(negative_path, token="easyneg")

    ip_model = IPAdapterPlusFT(pipe, image_encoder=image_encoder_path, controller_transforms=None, image_proj=ip_ckpt,
                               ip_adapter=ip_ckpt, t2i_adapter=ip_ckpt, device=device)

    if ip_model.t2i_adapter is not None:
        face_attr = facer.face_attr("farl/celeba/224", device=device)

    ip_model.to(device, torch.float16)

    demo = gr.Interface(partial(
        runCAMOUFLAGE,
        app=app,
        t=t,
        device=device,
        image_list=image_list,
        ip_model=ip_model,
        face_attr=face_attr,
    ), [
        gr.Image(type="pil"),
        gr.Slider(128, 2048, 768, step=64, label="Shape"),
        gr.Slider(0, 2, 1, step=0.01, label="Threshold"),
        gr.Slider(1, 100, 30, step=1, label="Steps"),
        gr.Slider(0, 1, 0.5, step=0.01, label="Alpha"),
        gr.Slider(0, 1, 0.55, step=0.01, label="Strength"),
        gr.Slider(0, 10, 3, step=0.5, label="Guidance"),
        gr.Slider(-1, 1000, -1, step=1, label="Seed"),
    ], ["image", gr.Text(container=False,show_label=False)], title="CAMOUFLAGE")
    demo.launch(share=True)


if __name__ == "__main__":
    main()
