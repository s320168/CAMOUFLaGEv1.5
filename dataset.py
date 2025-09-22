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
import cv2
import numpy as np
import os


class MyDataset(Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, tfms=None, controller_tfms=None,
                 use_t2i=False, pose_processor=None, use_triplets=False, max_boxes_per_data=8, embedding_len=768, token_len=77):
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
        if self.use_t2i and pose_processor is not None:
            self.pose_processor = pose_processor
            self.preprocessing()
        self.use_triplets = use_triplets
        self.max_boxes_per_data = max_boxes_per_data
        self.embedding_len = embedding_len
        self.token_len = token_len

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        # these entries would change based on the dataset passed through FRESCO and the chosen SGG model: for example the caption would be taken from
        # the extended scene graph obtained by processing FRESCO's identikit
        image_file = item["file_name"]

        # TODO
        # here the image should be passed to FRESCO to generate the corresponding identikit;
        # then should be passed to the SGG model to generate the triplets;
        # finally SGfy has to be called to generate the extended scene graph.

        # load the extended scene graph file in a dictionary
        with open("../dataset/FFHQ/extended_sg/" + image_file.split(".")[0] + ".json") as f:
            ext_sg = json.load(f)

        out = {}

        text = ext_sg["scene"]["single_action_caption"]

        # convert relationships triplets into the corresponding strings "subject relation object"
        triplets = ""
        if self.use_triplets:
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
                        if next_triplets.count(" ") + next_triplets.count(",")-1 + next_triplets.count("-")*2 <= self.token_len:
                            triplets = next_triplets
                        else:
                            hit_max_len = True
                        break
                if hit_max_len:
                    break
            triplets = triplets[:-2]

        # read image
        # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        raw_image = Image.open("../dataset/FFHQ/FFHQ-itw-512/" + image_file)
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.controller_transforms(images=raw_image, return_tensors="pt").pixel_values

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
            triplets = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            triplets = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        triplets_input_ids = self.tokenizer(
            triplets,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        res = None
        if self.use_t2i:
            raw_image = ((image / 2 + 0.5) * 255).unsqueeze(0)
            with torch.inference_mode():
                shape = image.shape[-1]
                res = get_datamaps(ext_sg, shape, shape, image_file)
                
        out["image"] = image
        out["text_input_ids"] = text_input_ids
        out["triplets_input_ids"] = triplets_input_ids
        out["clip_image"] = clip_image
        out["drop_image_embed"] = drop_image_embed
        out["facer"] = res

        return out

    def __len__(self):
        return len(self.data)

    def pose_preprocessing(self, img: np.ndarray, h: int, w: int, image_file: str):
        # get the pose estimation map from Openpose and convert it to numpy array
        openpose_image = self.pose_processor(np.uint8(img*255), include_hand=True, include_face=True, detect_resolution=img.shape[0], image_resolution=img.shape[0])
        open_cv_image = np.array(openpose_image)
        # convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        # resize image to the downsampled shape
        image = cv2.resize(open_cv_image, (h, w), interpolation=cv2.INTER_LINEAR)
        # get the greyscale version of the image
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # turn the greyscale image into a black and white one
        _, blackAndWhiteImage = cv2.threshold(grayImage, 1, 1, cv2.THRESH_BINARY)
        cv2.imwrite("../dataset/FFHQ/openpose/" + image_file, blackAndWhiteImage)

    def palette_preprocessing(self, img: np.array, h: int, w: int, image_file: str):
        # resize image to get a 1/8 downsample
        img_r = cv2.resize(img*255, (h, w), interpolation=cv2.INTER_LINEAR)
        Z = img_r.reshape((-1, 3))
        Z = np.float32(Z)
        # define criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # number of colors in the palette
        K = 8
        # apply kmeans
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        label_flat = label.flatten()
        # compose final output using color palette stored in center at the indexes stored in label_flat
        res = center[label_flat]
        # rehape to the resized image dimensions and bring values in [0, 1) range
        res = res.reshape(img_r.shape)
        cv2.imwrite("../dataset/FFHQ/palette/" + image_file, res)

    def preprocessing(self):
        h = w = self.size // 8
        for i in range(len(self.data)):
            item = self.data.iloc[i]
            image_file = item["file_name"]

            # read original image, convert it into RGB format and resize it into the needed shape
            raw_image = Image.open("../dataset/FFHQ/FFHQ-itw-512/" + image_file)
            tfms = transforms.Compose([
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])
            image = tfms(raw_image.convert("RGB"))
            img = image.numpy().transpose(1, 2, 0)
            if not os.path.isfile("../dataset/FFHQ/openpose/" + image_file):
                self.pose_preprocessing(img, h, w, image_file)
            if not os.path.isfile("../dataset/FFHQ/palette/" + image_file):
                self.palette_preprocessing(img, h, w, image_file)

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    triplets_input_ids = torch.cat([example["triplets_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    if data[0]["facer"] is None:
        facer = None
    else:
        facer = torch.cat([example["facer"] for example in data], dim=0)

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "triplets_input_ids": triplets_input_ids,
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