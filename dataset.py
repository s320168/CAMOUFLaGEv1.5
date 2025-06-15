import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import facer


class MyDataset(Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, tfms=None,
                 controller_tfms=None, use_t2i=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate

        self.data = pd.read_csv(json_file)

        if tfms is None:
            tfms = transforms.Compose([
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        self.transform = tfms

        if controller_tfms is None:
            controller_tfms = CLIPImageProcessor()
        self.controller_transforms = controller_tfms

        self.face_detector = None
        self.face_attr = None
        if use_t2i:
            self.face_detector = facer.face_detector("retinaface/mobilenet", device="cpu")
            self.face_attr = facer.face_attr("farl/celeba/224", device="cpu")

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item["caption"]
        image_file = item["file_name"]

        # read image
        # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        raw_image = Image.open(image_file)
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.controller_transforms(images=raw_image, return_tensors="pt").pixel_values

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        res = None
        if self.face_detector is not None:
            raw_image = ((image / 2 + 0.5) * 255).unsqueeze(0)
            with torch.inference_mode():
                shape = image.shape[-1]
                latent_shape = shape // 8
                res = torch.zeros((1, 41, latent_shape, latent_shape))
                faces = self.face_detector(raw_image)
                if "image_ids" not in faces:
                    print(f"No face detected - {image_file}")
                else:
                    faces = self.face_attr(raw_image, faces)
                    ids = faces["image_ids"].tolist()
                    rects = faces["rects"]
                    points = faces["points"]
                    attrs = faces["attrs"]  # TODO:Change fix size to variable size
                    # res = torch.zeros((len(set(ids)), 41, latent_shape, latent_shape))
                    for n, i in enumerate(ids):
                        p = torch.nn.functional.hardtanh(points[n] * (latent_shape - 1) // shape, 0,
                                                         latent_shape - 1).to(int)
                        # assert p.max().item() < 64
                        # assert p.min().item() >= 0
                        r = torch.nn.functional.hardtanh(rects[n] * latent_shape // shape, 0, latent_shape).to(int)
                        res[i, -1, p[:, 1], p[:, 0]] = 1
                        res[i, :40, max(r[1], 0):min(r[3], 64), max(r[0], 0):min(r[2], 64)] = attrs[n].repeat(
                            min(r[3], 64) - max(r[1], 0), min(r[2], 64) - max(r[0], 0), 1).permute(2, 0, 1)

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "facer": res
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    if data[0]["facer"] is None:
        facer = None
    else:
        facer = torch.cat([example["facer"] for example in data], dim=0)

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "facer": facer
    }


class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_path, transform):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms

        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        folder_path = Path(folder_path)
        self.image_list = list(folder_path.glob('*.*'))
        # Calculate len
        self.data_len = len(self.image_list)
        self.tfms = transform

    def __getitem__(self, index):
        single_image_path = self.image_list[index]
        im_as_im = Image.open(single_image_path).convert("RGB")
        return self.tfms(im_as_im), single_image_path.with_suffix("").name

    def __len__(self):
        return self.data_len
