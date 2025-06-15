import json
import os
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from torchvision import transforms
from tqdm import tqdm
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_type = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'
torch.set_default_tensor_type(tensor_type)
batch_size = 512


def images_to_grid(images, texts):
    dims = 256
    images = [np.array(Image.open(image).resize((dims, dims))) for image in images]
    w = round(np.sqrt(len(images)))
    h = int(np.ceil(len(images) / w))
    image = np.zeros((h * dims, w * dims, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        y = i // w
        x = i % w
        cv2.putText(img, f'{texts[i]}', (0, dims - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        image[y * dims:(y + 1) * dims, x * dims:(x + 1) * dims] = img
    return image


def dist_between_two_images(img1, img2, model, transforms):
    img1, img2 = Path(img1), Path(img2)
    img1 = img1.parent.parent / (img1.parent.name + "_facecrops") / img1.name
    img2 = img2.parent.parent / (img2.parent.name + "_facecrops") / img2.name
    img1 = transforms(Image.open(img1)).unsqueeze(0).to(device)
    img2 = transforms(Image.open(img2)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)
    return torch.dist(emb1, emb2)  # (emb1 - emb2).pow(2).sum().sqrt().cpu().numpy().squeeze().item()


def retrieve_top_k(input_path: str,
                   src_dataset: str,
                   ano_dataset: str,
                   k: int = 7):
    input_path = Path(input_path)
    output = input_path / 'imgs'

    # Create the output dir
    # output.mkdir(exist_ok=True, parents=True)

    for test in ["facenet_casia_webface", "facenet_vggface2", "clip"]:

        if "facenet" in test:
            model = InceptionResnetV1(pretrained='vggface2').eval().to(device) if "vgg" in test else InceptionResnetV1(
                pretrained='casia-webface').eval().to(device)

            facenet_transforms = transforms.Compose([np.float32,
                                                     transforms.ToTensor(),
                                                     fixed_image_standardization
                                                     ])

        for test in [test, f"{test}-emb"]:
            with open(input_path / f"{test}-map.json", "r") as f:
                nn_map = json.load(f)
            with open(input_path / f"{test}-map-dist.json", "r") as f:
                nn_map_dist = json.load(f)

            tmp_out = output / test
            tmp_out.mkdir(exist_ok=True, parents=True)

            # j = 0
            for idx in tqdm(list(nn_map.keys()), desc=f"Processing {test}"):
                # if j > 10:
                #     break
                # j += 1
                idx_original = idx.replace('_12345', "")
                idx_original = idx_original.replace('png', "jpg")
                # if any([idx_original != k for k in nn_map[idx]]):
                #     continue
                top_k = nn_map[idx][:k]
                # replace using regex in idx_original
                tmp = re.sub(r'_[0-9]+.jpg', '', idx_original)
                if not any([tmp == re.sub(r'_[0-9]+.jpg', '', k) for k in top_k]):
                    continue
                top_k_dist = nn_map_dist[idx][:k]
                imgs = [f"{src_dataset}/{idx_original}", f"{ano_dataset}/{idx}"]
                texts = [f"{idx_original}-{0}",
                         f"{idx}-?-{dist_between_two_images(imgs[0], imgs[1], model, facenet_transforms):.3f}"]
                for i, img_file in enumerate(top_k):
                    img_path = f'{src_dataset}/{img_file}'
                    # If not found, check in a subfolder
                    if not os.path.isfile(img_path):
                        subfolder = '_'.join(img_file.split('.')[0].split('_')[:-1])
                        img_path = f"{src_dataset}/{subfolder}/{img_file}"
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(f'The file "{idx}" does not exist.')
                    img_file = img_file.split(".")[0]

                    # Read the image
                    # img = cv2.imread(img_path)
                    texts.append(
                        f"{img_file}-{top_k_dist[i]:.3f}-{dist_between_two_images(imgs[0], img_path, model, facenet_transforms):.3f}")
                    imgs.append(img_path)

                # Concat and save
                Image.fromarray(images_to_grid(imgs, texts)).save(
                    tmp_out / f"{test}_{idx_original.split('.')[0]}-top_{k}.jpg")
                # cv2.imwrite(f"./output/imgs/{test}_{idx.split('.')[0]}-top_{k}.jpg", im_h)


@click.command()
@click.option('--input_path')
@click.option('--src_dataset')
@click.option('--ano_dataset')
@click.option('--k', default=7)
def main(input_path, src_dataset, ano_dataset, k):
    #Check inputs
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path {input_path} is not a valid directory")
    if not os.path.isdir(src_dataset):
        raise ValueError(f"Source dataset path {src_dataset} is not a valid directory")
    if not os.path.isdir(ano_dataset):
        raise ValueError(f"Anomaly dataset path {ano_dataset} is not a valid directory")
    if k < 1:
        raise ValueError(f"K must be greater than 0")
    retrieve_top_k(input_path, src_dataset, ano_dataset, k)


if __name__ == '__main__':
    main()
