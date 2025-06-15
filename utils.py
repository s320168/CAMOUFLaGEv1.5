import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from ip_adapter import IPAdapterPlus, IPAdapterPlusXL
from ip_adapter.utils import FacerAdapter


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x, *args, **kwargs):
        return self.lambd(x)


def set_requires_grad(requires_grad=False, *args):
    for param in args:
        param.requires_grad_(requires_grad)


def set_device_dtype(device, dtype, *args):
    for param in args:
        param.to(device, dtype=dtype)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_adapter_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )

    parser.add_argument(
        "--use_t2i",
        action="store_true",
        help=(
            "Whether or not to use t2i adapter"
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")

    parser.add_argument("--usev2", action="store_true", help="usev2")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class IPAdapterPlusXLFT(IPAdapterPlusXL):
    def __init__(self, pipe, image_encoder, controller_transforms, image_proj=None, ip_adapter=None, t2i_adapter=None,
                 device=None):
        self.pipe = pipe
        self.device = device
        # load image encoder
        if isinstance(image_encoder, (Path, str)):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder).to(dtype=torch.float16)
        else:
            self.image_encoder = image_encoder
        self.clip_image_processor = controller_transforms if controller_transforms is not None else CLIPImageProcessor()
        # image proj model
        if isinstance(image_proj, (Path, str)):
            self.num_tokens = 16
            state_dict = torch.load(image_proj, map_location="cpu")
            self.image_proj_model = self.init_proj()
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
        else:
            self.image_proj_model = image_proj

        if ip_adapter is not None:
            self.set_ip_adapter()
            state_dict = torch.load(ip_adapter, map_location="cpu")
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])

        if isinstance(t2i_adapter, (Path, str)):
            self.t2i_adapter = FacerAdapter()
            self.t2i_adapter.load_state_dict(torch.load(t2i_adapter, map_location="cpu")["t2i_adapter"])
        elif t2i_adapter is not None:
            self.t2i_adapter = t2i_adapter

    def to(self, device, dtype=None):
        if device is not None:
            self.image_encoder.to(device)
            self.image_proj_model.to(device)
            self.pipe.to(device)
            if self.t2i_adapter is not None:
                self.t2i_adapter.to(device)
        if dtype is not None:
            self.image_encoder.to(dtype=dtype)
            self.image_proj_model.to(dtype=dtype)
            if self.t2i_adapter is not None:
                self.t2i_adapter.to(dtype=dtype)
        return self


class IPAdapterPlusFT(IPAdapterPlus):
    def __init__(self, pipe, image_encoder, controller_transforms, image_proj=None, ip_adapter=None, t2i_adapter=None,
                 device=None):
        self.pipe = pipe
        self.device = device
        # load image encoder
        if isinstance(image_encoder, (str, Path)):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder).to(dtype=torch.float16)
        else:
            self.image_encoder = image_encoder
        self.clip_image_processor = controller_transforms if controller_transforms is not None else CLIPImageProcessor()
        # image proj model
        if isinstance(image_proj, (str, Path)):
            self.num_tokens = 16
            state_dict = torch.load(image_proj, map_location="cpu")
            self.image_proj_model = self.init_proj()
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
        else:
            self.image_proj_model = image_proj

        if ip_adapter is not None:
            self.set_ip_adapter()
            state_dict = torch.load(ip_adapter, map_location="cpu")
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])

        if isinstance(t2i_adapter, (str, Path)):
            state_dict = torch.load(t2i_adapter, map_location="cpu")
            if "t2i_adapter" in state_dict and len(state_dict["t2i_adapter"]) != 0:
                self.t2i_adapter = FacerAdapter()
                self.t2i_adapter.load_state_dict(state_dict["t2i_adapter"])
            else:
                self.t2i_adapter = None
                del state_dict
        elif t2i_adapter is not None:
            self.t2i_adapter = t2i_adapter
        else:
            self.t2i_adapter = None

    def to(self, device, dtype=None):
        if device is not None:
            self.image_encoder.to(device)
            self.image_proj_model.to(device)
            self.pipe.to(device)
            if self.t2i_adapter is not None:
                self.t2i_adapter.to(device)
        if dtype is not None:
            self.image_encoder.to(dtype=dtype)
            self.image_proj_model.to(dtype=dtype)
            if self.t2i_adapter is not None:
                self.t2i_adapter.to(dtype=dtype)
        return self


def draw_image_landmarks(image, points, color=(255, 255, 0)):
    for i, point in enumerate(points):
        cv2.circle(image, (int(point[0]), int(point[1])), 2, color, -1)
    return image


def draw_image_landmarks_name(image, points, color=(255, 255, 0)):
    for i, point in enumerate(points):
        cv2.putText(image, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return image


def draw_image_landmarks_lines(image, points):
    for i in range(0, len(points) - 1):
        cv2.line(image, (int(points[i][0]), int(points[i][1])), (int(points[i + 1][0]), int(points[i + 1][1])),
                 (0, 255, 0), 1)
    # cv2.line(image, (int(points[-1][0]), int(points[-1][1])), (int(points[0][0]), int(points[0][1])), (0, 255, 0), 1)
    return image


# get center of points
def center(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.mean(x), np.mean(y)


def scale_points(points: list, scale=1.2, move=(0, 0)):
    points = np.array(points)
    pos = np.array(center(*points.T))
    new_a = (points - pos) * scale + pos
    new_a += np.array(move)
    return new_a.tolist()


def get_landmarks_points106(face):
    return [(int(x[0]), int(x[1])) for x in face.landmark_2d_106.astype(np.int32)]


def get_landmarks_points68(face, is_2d=True):
    points = []
    for x in face.landmark_3d_68.astype(np.int32):
        points.append((int(x[0]), int(x[1]), int(x[2])))
        if is_2d:
            points[-1] = points[-1][:2]

    return points


def maybe_int(s):
    if s.isdigit():
        return int(s)
    return s


def images_to_grid(images, texts=None):
    if len(images) == 0:
        return None
    images = [np.array(Image.open(image) if isinstance(image, (str, Path)) else image) for image in images]
    w = int(np.sqrt(len(images)))
    h = int(np.ceil(len(images) / w))
    if w < h:
        w, h = h, w
    image = np.zeros((h * images[0].shape[0], w * images[0].shape[1], 3), dtype=np.uint8)
    for i, img in enumerate(images):
        y = i // w
        x = i % w
        if texts is not None:
            cv2.putText(img, f'{texts[i]}', (0, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2,
                        cv2.LINE_AA)
        image[y * img.shape[0]:(y + 1) * img.shape[0], x * img.shape[1]:(x + 1) * img.shape[1]] = img
    return image


def get_concat_h(*args):
    dst = Image.new('RGB', (sum([x.width for x in args]), args[0].height))
    x_offset = 0
    for im in args:
        dst.paste(im, (x_offset, 0))
        x_offset += im.width
    return dst


def get_concat_v(*args):
    dst = Image.new('RGB', (args[0].width, sum([x.height for x in args])))
    y_offset = 0
    for im in args:
        dst.paste(im, (0, y_offset))
        y_offset += im.height
    return dst


def get_face_direction(landmarks_points, is_68=False):
    if not is_68:
        nose = landmarks_points[86][0]
        eye1 = landmarks_points[34][0] - nose
        eye2 = landmarks_points[88][0] - nose
    else:
        nose = landmarks_points[30][0]
        eye1 = landmarks_points[36][0] - nose
        eye2 = landmarks_points[45][0] - nose

    # if eye2 < 0:
    #     return 1
    # elif eye1 > 0:
    #     return -1
    if abs(eye1) > abs(eye2):
        return 1
    else:
        return -1


class catchtime(object):
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t
        print(self.t)
