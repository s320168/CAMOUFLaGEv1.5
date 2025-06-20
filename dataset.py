import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
from utils import get_datamaps
import json


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
        self.use_t2i = use_t2i

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        # these entries would change based on the dataset passed through FRESCO and the chosen SGG model: for example the caption would be taken from
        # the extended scene graph obtained by processing FRESCO's identikit
        image_file = item[0]

        # TODO
        # here the image should be passed to FRESCO to generate the corresponding identikit;
        # then should be passed to the SGG model to generate the triplets;
        # finally SGfy has to be called to generate the extended scene graph.

        # load the extended scene graph file in a dictionary
        with open("data/input/extended_sg/extended_sg_" + image_file.split(".")[0] + ".json") as f:
            ext_sg = json.load(f)

        text = ext_sg["scene"]["single_action_caption"]

        # read image
        # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        raw_image = Image.open("data/input/images/" + image_file)
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
        if self.use_t2i is not None:
            raw_image = ((image / 2 + 0.5) * 255).unsqueeze(0)
            with torch.inference_mode():
                shape = image.shape[-1]
                latent_shape = shape // 8
                res = get_datamaps(ext_sg, shape, shape, image_file)

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
