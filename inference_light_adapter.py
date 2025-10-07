import os
os.environ['HF_HOME'] = 'E:/Tesi_Silvano/weights/'

import click
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL
from torchvision.transforms.functional import to_tensor

from face_swap import face_swap
from ip_adapter.utils import is_torch2_available
from pipelines import StableDiffusionImg2ImgPipelineRes
from utils import IPAdapterPlusFT, maybe_int, get_concat_h, images_to_grid, AttributeDict, get_datamaps

if is_torch2_available():
    from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor
else:
    from diffusers.models.attention_processor import AttnProcessor

from pathlib import Path

import facer
import torchvision
from tqdm import tqdm

from nearest_face import AnnoyWrapper

from PIL import Image
#from insightface.app import FaceAnalysis
from wakepy import keep
import json


def write_crop(image, crop_path, face, embed_path):
    face = {key: value for key, value in face.items() if key != "embedding"}
    np.save(embed_path, face)
    Image.fromarray(image).crop(face["bbox"]).save(crop_path)


@click.command()
@click.option('--crop_path', type=click.Path(), help='Path to folder with crops')
@click.option('--method', type=str, help='Method to use for embedding',
              default="CLIP:Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64")
@click.option('--file_index', type=click.Path(exists=False), help='Path to file index', default="farl.ann")
@click.option('--batch_size', type=int, help='Batch size', default=128)
@click.option('--base_model_path', type=str, help='Path to base model',
              default="stablediffusionapi/realistic-vision-v51")
@click.option('--vae_model_path', type=str, help='Path to VAE model', default="stabilityai/sd-vae-ft-mse")
@click.option('--image_encoder_path', type=str, help='Path to image encoder',
              default="Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64")
@click.option('--ip_ckpt', type=str, help='Path to IP checkpoint', default="light_1.5v.bin")
@click.option('--device', type=str, help='Device', default="cuda" if torch.cuda.is_available() else "cpu")
@click.option('--output_folder', type=click.Path(exists=False), help='Path to output folder')
@click.option('--in_folder', type=click.Path(exists=True), help='Path to input folder')
@click.option('--negative_path', type=click.Path(exists=True), help='Path to negative images')
@click.option('--skip_t2i', is_flag=True, help='Skip T2I')
@click.option('--seed', type=int, help='Seed', default=12345)
@click.option('--strength', type=float, help='Strength', default=0.55)
@click.option('--alpha', type=float, help='Scale', default=0.5)
@click.option('--max_size', type=int, help='Max size', default=768)
@click.option('--min_threshold', type=float, help='Min threshold', default=1)
@click.option('--num_inference_steps', type=int, help='Num inference steps', default=30)
@click.option("--lb", type=int, help="Lower bound", default=0)
@click.option("--no_seamless_copy", is_flag=True, help="Disable seamless copy")
@click.option("--output_verbose", is_flag=True, help="Output verbose")
@click.option('--cfg', type=float, help='Guidance scale', default=3)
@click.option("--usev2", is_flag=True, help="Use Resampler2 for image projection")
def main(crop_path, method, file_index, batch_size, base_model_path, vae_model_path, image_encoder_path,
         ip_ckpt, device, output_folder, in_folder, negative_path, skip_t2i, seed, strength, alpha, max_size,
         min_threshold, num_inference_steps, lb, no_seamless_copy, output_verbose, cfg, usev2):
    original_path = Path(crop_path)
    file_index = Path(file_index)
    output_folder = Path(output_folder)
    negative_path = Path(negative_path)
    ip_ckpt = Path(ip_ckpt)
    in_folder = Path(in_folder)

    #app = FaceAnalysis(name='buffalo_l')
    #app.prepare(ctx_id=0, det_size=(512, 512))

    output_folder.mkdir(parents=True, exist_ok=True)

    assert ip_ckpt.exists(), "IP checkpoint not found"
    assert in_folder.exists(), "Input folder not found"

    crop_path = original_path.with_name(f"{original_path.name}_crop")
    embeddings_path = original_path.with_name(f"{original_path.name}_embeddings")
    crop_path.mkdir(parents=True, exist_ok=True)
    embeddings_path.mkdir(parents=True, exist_ok=True)

    # image_list = [p for p in original_path.glob('*.*') if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    # assert len(image_list) > 0, "No images found"

    # for path in tqdm(image_list, desc="Checking cropped images..."):
    #     p_crop = crop_path / f"{path.stem}.png"
    #     p_emb = embeddings_path / f"{path.stem}.npy"
    #     if p_crop.exists() and p_emb.exists():
    #         continue
    #     image = np.array(Image.open(path).convert("RGB"))
    #     face = app.get(image, 1)
    #     if len(face) == 0:
    #         tqdm.write(f"Face not found in {path}")
    #         path.unlink()
    #     else:
    #         write_crop(image, p_crop, face[0], p_emb)

    t = AnnoyWrapper(method, device)

    if file_index.exists():
        t.annoy.load(str(file_index))
        print("Index loaded")
    else:
        print("Index not found! Generating...")
        t.generate(crop_path, batch_size=batch_size, name=str(file_index))
        print("Index generated")

    # load VAE
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load SD pipeline
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
    pipe.unet.set_attn_processor(AttnProcessor())

    if negative_path is not None and negative_path.exists():
        pipe.load_textual_inversion(negative_path, token="easyneg")

    ip_model = IPAdapterPlusFT(pipe, image_encoder=image_encoder_path, controller_transforms=None, image_proj=ip_ckpt,
                               ip_adapter=ip_ckpt, t2i_adapter=ip_ckpt if not skip_t2i else None, device=device, usev2=usev2)

    ip_model.to(device, torch.float16)

    img_paths = list(f for f in Path(in_folder).glob("**/*.*"))
    # sort paths by name
    img_paths = sorted(img_paths, key=lambda x: maybe_int(x.with_suffix("").name))
    img_paths = img_paths[lb:]

    # to_pil = torchvision.transforms.ToPILImage()
    tfms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(max_size),
        torchvision.transforms.CenterCrop((max_size, max_size)),
    ])
    # if ip_model.t2i_adapter is not None:
    #     labels = face_attr.labels

    for path in tqdm(img_paths):
        out_name = path.with_suffix("").name
        if (output_folder / f"{out_name}_{seed}.png").exists():
            tqdm.write(f"Skipping {path}, already exists")
            continue
        image = Image.open(path).convert("RGB")
        image = tfms(image)
        # faces = app.get(np.array(image))
        # faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        new_image = np.array(image.copy())
        face_used = []
        # blur_image = image.copy()
        # if len(faces) == 0:
        #     res = None
        # else:
        #     for i, face in enumerate(faces):
        #         rect = face.bbox.tolist()
        #         if (rect[2] - rect[0]) * (rect[3] - rect[1]) < 64 * 64:
        #             continue
        #         face_im = image.crop(rect)
        #         new_face = t.get_nns_by_vector(face_im, n=20, search_k=-1, include_distances=True)
        #         threshold = min(max(new_face[0]), min_threshold)
        #         new_face = list(zip(*new_face))
        #         new_faceb = [x for x in new_face if x[1] > threshold]
        #         new_face.reverse()
        #         new_face = (new_faceb if len(new_faceb) != 0 else new_face)[0][0]

        #         path = image_list[new_face]
        #         new_face = AttributeDict(
        #             np.load(path.parent.with_name(f"{path.parent.name}_embeddings") / f"{path.stem}.npy",
        #                     allow_pickle=True).item())
        #         fim = np.array(Image.open(path).convert("RGB"))
        #         face_used.append(path)

        #         new_image = face_swap(new_image, face, fim, new_face, no_seamless_copy)

        new_image = Image.fromarray(new_image)
        with open("../dataset/FFHQ/extended_sg/val/" + path.parts[-1].split(".")[0] + ".json") as f:
            ext_sg = json.load(f)

        if ip_model.t2i_adapter is None:
            res = None
        else:
            raw_image = to_tensor(new_image).unsqueeze(0).to(device) * 255
            h, w = raw_image.shape[-2:]
            latent_shape_h, latent_shape_w = h // 8, w // 8
            latent_max = max(latent_shape_h, latent_shape_w)
            shape_max = max(h, w)
            # torch.save(attrs, out / "attrs.pt")
            res = get_datamaps(ext_sg, shape_max, shape_max, os.path.join(Path("val/"), path.parts[-1]))

        if isinstance(new_image, np.ndarray):
            new_image = Image.fromarray(new_image)
        prompt_caption = ext_sg["scene"]["single_action_caption"]
        triplets = ""
        hit_max_len = False
        for rel in ext_sg["relationships"]:
            subj_hit = False
            obj_hit = False
            for obj in ext_sg["objects"]:
                if obj["id"] == rel["source"] and not subj_hit:
                    subj_hit = True
                    subject = obj["type"]
                elif obj["id"] == rel["target"] and not obj_hit:
                    obj_hit = True
                    object = obj["type"]
                if subj_hit and obj_hit:
                    next_triplets = triplets + f"{subject} {rel["type"]} {object}, "
                    if next_triplets.count(" ") + next_triplets.count(",")-1 + next_triplets.count("-")*2 <= 77:
                        triplets = next_triplets
                    else:
                        hit_max_len = True
                    break
            if hit_max_len:
                break
        prompt_triplets = triplets[:-2]

        images = ip_model.generate(pil_image=new_image, num_samples=1, num_inference_steps=num_inference_steps,
                                   seed=seed,
                                   guidance_scale=cfg,
                                   prompt=prompt_caption,
                                   prompt_triplets=prompt_triplets,
                                   negative_prompt="easyneg", 
                                   image=Image.blend(new_image, image, alpha), strength=strength, scale=1,
                                   down_block_additional_residuals=None if res is None else ip_model.t2i_adapter(
                                       res.to(device, torch.float16)))

        for i, img in enumerate(images):
            if output_verbose:
                get_concat_h(image, new_image, img,
                             Image.fromarray(images_to_grid(face_used)).resize(image.size)).save(
                    output_folder / f"{out_name}_{i}_new.png")
            img.save(output_folder / f"{out_name}_{seed + i}_str_{strength}_cfg_{cfg}.png")


if __name__ == '__main__':
    with keep.running() as k:
        main()
