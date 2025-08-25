import itertools
import os
os.environ['HF_HOME'] = 'E:/Tesi_Silvano/weights/'
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

from dataset import MyDataset, collate_fn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.utils import is_wandb_available
from ip_adapter.utils import is_torch2_available, init_ip_adapter, FacerAdapter
from pipelines import StableDiffusionImg2ImgPipelineRes
from utils import parse_args, set_requires_grad, set_device_dtype, IPAdapterPlusFT, get_datamaps
from controlnet_aux import OpenposeDetector

if is_torch2_available():
    from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor
else:
    from diffusers.models.attention_processor import AttnProcessor

if is_wandb_available():
    import wandb

import torchvision.transforms.functional as TF
from torchvision import transforms

with open("data/wandb.txt", "r") as f:
    k = f.read()
wandb.login(key=k.strip())

logger = get_logger(__name__)


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    if args.use_gligen:
        # define custom UNetConditionedModel
        print("GLIGEN not yet implemented")
    else:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        unet.set_attn_processor(AttnProcessor())
        grounding_tokenizer_input = None
    if args.use_farl:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # here would be added FRESCO and the selected SGG model loadings

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )
    # freeze parameters of models to save more memory
    set_requires_grad(False, vae, unet, text_encoder)
    if args.use_farl:
        set_requires_grad(False, image_encoder)

    # ip-adapter-plus
    ip_adapter = init_ip_adapter(num_tokens=16, unet=unet, image_encoder=image_encoder if args.use_farl else None, 
                                 usev2=args.usev2, t2i_adapter=FacerAdapter() if args.use_t2i else None)

    # define OpenPose model and transfer it on the device used
    if args.use_t2i is not None:
        openpose_processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(accelerator.device)

    if args.load_adapter_path is not None:
        ip_adapter.load_state_dict(torch.load(args.load_adapter_path), strict=False)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # set_device_dtype(accelerator.device, weight_dtype, vae, unet, text_encoder, image_encoder, ip_adapter)

    vae, unet, text_encoder, ip_adapter = accelerator.prepare(
        vae, unet, text_encoder, ip_adapter
    )
    if args.use_farl:
        image_encoder = accelerator.prepare(image_encoder) 

    # optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(ip_adapter.image_proj.parameters() if args.use_farl else [], ip_adapter.ip_adapter.parameters(),
                        ip_adapter.t2i_adapter.parameters() if ip_adapter.t2i_adapter is not None else []),
        lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    dataset = MyDataset(args.data_file, tokenizer=tokenizer, size=args.resolution, use_t2i=args.use_t2i,
                        controller_tfms=CLIPImageProcessor(args.image_encoder_path), 
                        pose_processor=openpose_processor if args.use_t2i is not None else None,
                        use_gligen=args.use_gligen)

    tfms, controller_transforms = dataset.transform, dataset.controller_transforms
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)


    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    max_train_steps = args.num_train_epochs * len(train_dataloader.dataset)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) ="
        f" {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(0, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(
                        batch["images"].to(accelerator.device)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset > 0:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(
                        accelerator.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.use_farl:
                    clip_images = []
                    for clip_image, drop_image_embed in zip(batch["clip_images"], batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            clip_images.append(torch.zeros_like(clip_image))
                        else:
                            clip_images.append(clip_image)
                    clip_images = torch.stack(clip_images, dim=0)
                    with torch.no_grad():
                        image_embeds = image_encoder(clip_images,
                                                    output_hidden_states=True).hidden_states[-2]

                with torch.no_grad():
                    encoder_hidden_states_caption = text_encoder(batch["text_input_ids"])[0]
                    if args.use_triplets:
                        encoder_hidden_states_triplets = text_encoder(batch["triplets_input_ids"])[0]
                    else:
                        encoder_hidden_states_triplets = None

                grounding_input = None
                if args.use_t2i:
                    image_embeds2 = batch["facer"].to(accelerator.device)
                    if args.use_gligen:
                        grounding_input = grounding_tokenizer_input.prepare(batch)
                else:
                    image_embeds2 = None

                noise_pred = ip_adapter(unet, noisy_latents, timesteps, encoder_hidden_states_caption, 
                                        encoder_hidden_states_triplets, image_embeds if args.use_farl else None, 
                                        image_embeds2=image_embeds2, grounding_input=grounding_input)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(ip_adapter.parameters(), 1.)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            save_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                            image_logs = log_validation(
                                vae,
                                text_encoder,
                                tokenizer,
                                unet,
                                tfms,
                                controller_transforms,
                                image_encoder if args.use_farl else None,
                                ip_adapter,
                                args,
                                accelerator,
                                accelerator.mixed_precision,
                                global_step
                            )

                logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

    accelerator.end_training()


def log_validation(vae, text_encoder, tokenizer, unet, tfms, controller_transforms, image_encoder, ipAdapterTrainer,
                   args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    # controlnet = accelerator.unwrap_model(cunet)
    pipeline = StableDiffusionImg2ImgPipelineRes.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                                 algorithm_type="sde-dpmsolver++",
                                                                 use_karras_sigmas=True)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    token = "<easyneg>"
    if token in tokenizer.get_vocab():
        pipeline.load_textual_inversion("embeddings/easynegative.safetensors",
                                        token=token)

    # %%

    ip_model = IPAdapterPlusFT(pipeline, image_encoder, controller_transforms, ipAdapterTrainer.image_proj)
    ip_model.device = accelerator.device

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    tfms = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        image_file = validation_image
        raw_image = Image.open("data/input/images/" + image_file)
        image = tfms(raw_image.convert("RGB"))
        validation_image = TF.to_pil_image(image, mode="RGB")
        validation_prompt = validation_prompt.split(".")
        if args.use_triplets and len(validation_prompt[1]) > 0:
            validation_prompt[1] = validation_prompt[1][1:]
        else:
            validation_prompt[1] = None

        images = []
        with torch.autocast("cuda"), torch.no_grad():
            with torch.inference_mode():
                if ipAdapterTrainer.t2i_adapter is None:
                    res = None
                else:
                    raw_image = ((tfms(validation_image) / 2 + 0.5) * 255).unsqueeze(0).to(accelerator.device)
                    shape = raw_image.shape[-1]
                    latent_shape = shape // 8
                    # TODO: pass image to FRESCO and to the SGG model

                    # load the extended scene graph file in a dictionary
                    with open("data/input/extended_sg/extended_sg_" + image_file.split(".")[0] + ".json") as f:
                        ext_sg = json.load(f)
                    res = get_datamaps(ext_sg, shape, shape, image_file)

            # using crop_image instead of validation_image in the arguments for the reason above
            tmp = ip_model.generate(pil_image=validation_image, num_samples=args.num_validation_images,
                                    num_inference_steps=30, prompt=validation_prompt[0], prompt_triplets=validation_prompt[1], 
                                    seed=args.seed, negative_prompt=token, image=validation_image, strength=0.6,
                                    scale=0.8 if len(validation_prompt) > 1 else 1.0,
                                    down_block_additional_residuals=None if res is None else ipAdapterTrainer.t2i_adapter(
                                        res.to(accelerator.device)))

        images.extend(tmp)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


if __name__ == "__main__":
    main()
