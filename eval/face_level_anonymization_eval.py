import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torchvision
from PIL import Image
from clip import clip
from facenet_pytorch import MTCNN, training
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import nearest_face
from dataset import CustomDatasetFromFile
from generic_dataset import DatasetImgFolder
from ir_metrics import mean_average_precision_at_k, mean_recall_at_k
import numpy as np

parser = argparse.ArgumentParser(description="Face-level anonymization evaluation.")
parser.add_argument('--real_dataset', help="root directory of the real dataset", type=str, required=True)
parser.add_argument('--anonymized_dataset', help="root directory of the real dataset", type=str, required=True)

parser.add_argument('--neighbors', help="number of neighbors to be retrieved", type=int, default=50)

args = parser.parse_args()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_type = "torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor"
torch.set_default_tensor_type(tensor_type)
# torch.set_default_dtype(tensor_type)
batch_size = 512


def extract_embeddings(imgs, model, transform, out_path):
    dataset = CustomDatasetFromFile(imgs, transform)
    print(f"Extract: Found {len(dataset)} images.")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8)

    with torch.no_grad():
        embds = None
        for batch in tqdm(dataloader):
            images, _ = batch
            out = model(images, device).cpu()
            if embds is None:
                embds = out
            else:
                embds = torch.cat([embds, out], dim=0)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(embds, out_path)


def create_crops(path, crop_path, file_index=None):
    if file_index is not None and file_index.exists():
        file_index.unlink()

    mtcnn = MTCNN(image_size=160,
                  margin=0,
                  select_largest=True,
                  keep_all=False,
                  device=device)

    dataset = DatasetImgFolder(root=str(path), transform=None, out_crop=crop_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                            collate_fn=training.collate_pil)
    for images, images_paths in tqdm(dataloader, desc="Extracting faces with MTCNN: "):
        crops_out_paths = [p.replace(str(path), str(crop_path)) for p in images_paths]
        res = list(zip(*[(p, img) for p, img in zip(crops_out_paths, images) if not Path(p).exists()]))
        if len(res) == 0:
            continue
        crops_out_paths, images = res

        # Extract faces and save images
        with torch.no_grad():
            mtcnn(images, save_path=crops_out_paths)

            for p in crops_out_paths:
                if not Path(p).exists():
                    print(f"WARNING: {p} not found!")
                    Image.fromarray(np.zeros((160, 160, 3), dtype=np.uint8)).save(p)


def find_neighbors(anonymized_path,
                   original_path,
                   test='facenet_vggface2',
                   k_sample=50,
                   skip_aprox=True):
    # Check if NN map file already exists

    anonymized_path = Path(anonymized_path)
    original_path = Path(original_path)
    out_map = Path(f'./') / anonymized_path.name / f'{test}-map.json'
    out_map_dist = out_map.parent / f'{test}-map-dist.json'
    out_map_emb = out_map.parent / f'{test}-emb-map.json'
    out_map_emb_dist = out_map.parent / f'{test}-emb-map-dist.json'

    out_map.parent.mkdir(exist_ok=True, parents=True)

    print(f"Computing for {test}... OutIn: {out_map.parent}")

    if "clip" not in test:
        crop_path = original_path.parent / f'{original_path.name}_facecrops'
    else:
        crop_path = original_path
    if "clip" not in test:
        crop_path_anonymized = anonymized_path.parent / f'{anonymized_path.name}_facecrops'
    else:
        crop_path_anonymized = anonymized_path
    file_index = original_path.parent / f'{test}-index.ann'
    embeddings_path = original_path.parent / f'{original_path.name}_embeddings' / f'{test}_embeds.pt'

    if "clip" not in test:
        create_crops(original_path, crop_path, file_index)
        create_crops(anonymized_path, crop_path_anonymized)

    if not anonymized_path.exists():
        raise FileNotFoundError(f'The path "{anonymized_path}" does not exist.')
    if not original_path.exists():
        raise FileNotFoundError(f'The path "{original_path}" does not exist.')

    if not skip_aprox:
        if out_map.exists():
            print(f'NN Map file already available in {test}-map.json.'
                  f' Consider to delete this file to get updated results.')
        else:
            print("NN Map file not found! Generating...")
            t = nearest_face.AnnoyWrapper(test, device)
            nn_map = dict()
            nn_map_dist = dict()
            if file_index.exists():
                t.annoy.load(str(file_index))
                print("Index loaded")
            else:
                print("Index not found! Generating...")
                t.generate(crop_path, batch_size=batch_size, name=str(file_index))
                print("Index generated")

            crop_pathb = list(crop_path.rglob('*.*'))

            # Build NN map dicts

            for file in tqdm(list(crop_path_anonymized.rglob('*.*')), desc="Finding NNs: "):
                indices, distances = t.get_nns_by_vector(file, n=k_sample, search_k=-1, include_distances=True)
                nn_map[file.name] = [crop_pathb[int(idx)].name for idx in indices]
                nn_map_dist[file.name] = distances

            # Save NN map dicts
            with open(out_map, 'w') as f:
                json.dump(nn_map, f, indent=2)
            with open(out_map_dist, 'w') as f:
                json.dump(nn_map_dist, f, indent=2)

    if out_map_emb.exists():
        print(f'Embedding NN Map file already available in {test}-map.json.'
              f' Consider to delete this file to get updated results.')
    else:
        print("NN Embeddings file not found! Generating...")
        nn_map = dict()
        nn_map_dist = dict()
        if "face" in test:
            if "vgg" in test:
                method = "vggface2"
            elif "casia" in test:
                method = "casia-webface"
            else:
                raise NotImplementedError()
            m = InceptionResnetV1(pretrained=method).to(device).eval()
            model = lambda x, device, dtype=None: m(x.to(device, dtype=dtype))
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((160, 160)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # lambda x: x-0.5
            ])
        elif "CLIP" in test:
            method = test.split(":")[1]
            m = CLIPVisionModelWithProjection.from_pretrained(method).to(device, dtype=torch.float16).eval()
            model = lambda x, device: m(x.pixel_values.squeeze(1).to(device)).image_embeds
            transform = partial(CLIPImageProcessor(), return_tensors="pt")
        elif "clip" in test:
            m, transform = clip.load("ViT-B/32", device=device, jit=False)
            m.float()
            m.eval()
            model = lambda x, device: m.encode_image(x.to(device))
        else:
            raise NotImplementedError()
        if not embeddings_path.exists():
            print(f'Embeddings file not found in {embeddings_path}.')
            print(f'Extracting embeddings from {original_path}...')
            extract_embeddings(crop_path, model, transform, embeddings_path)
        else:
            print(f'Embeddings file already available in {embeddings_path}. Loading...')
        embeddings = torch.load(embeddings_path).to(device)

        crop_path = list(crop_path.rglob('*.*'))
        dataset = CustomDatasetFromFile(crop_path_anonymized, transform)
        crop_path_anonymized = [f.name for f in crop_path_anonymized.rglob('*.*')]
        print(f"FIND: Found {len(dataset)} images")
        dataloader = DataLoader(dataset, batch_size=1, num_workers=8)
        with torch.no_grad():
            for i, img in enumerate(tqdm(dataloader, desc="Finding emb NNs: ")):
                img = model(img[0], device)
                img = img.expand_as(embeddings)
                img = (img - embeddings).pow(2).sum(1).sqrt().cpu().topk(k_sample, largest=False)
                indices, distances = img.indices.squeeze().tolist(), img.values.squeeze().tolist()
                name = crop_path_anonymized[i]
                nn_map[name] = [crop_path[int(idx)].name for idx in indices]
                nn_map_dist[name] = distances

        with open(out_map_emb, 'w') as f:
            json.dump(nn_map, f, indent=2)
        with open(out_map_emb_dist, 'w') as f:
            json.dump(nn_map_dist, f, indent=2)

    out_map.parent.mkdir(exist_ok=True, parents=True)


def compute_ir_metrics(path, test, files_path):
    # Check if NN map file and original filenames file exist
    print(f'Computing IR metrics for {test}...')
    nn_map_path = path / f'{test}-map.json'
    out_file = nn_map_path.parent / f'{test}-ir_results.json'
    if out_file.exists():
        print(f"Already computed IR metrics for {test}.")
        print("Consider to delete this file to get updated results.")
        return

    ids = None
    if files_path.with_suffix(".json").exists():
        print(f'Compute: Found ids file for {test}...')
        with open(files_path.with_suffix(".json"), "r") as f:
            ids = json.load(f)

        # Load NN map file
    with open(nn_map_path, "r") as f:
        nn_map = json.load(f)
    # Load filenames
    all_imgs = [f.name for f in files_path.rglob('*.*')]

    # Compute ir metrics
    ir_results = {
        "Recall@1": mean_recall_at_k(nn_map, all_imgs, k=1, ids=ids),
        "Recall@3": mean_recall_at_k(nn_map, all_imgs, k=3, ids=ids),
        "Recall@5": mean_recall_at_k(nn_map, all_imgs, k=5, ids=ids),
        "Recall@10": mean_recall_at_k(nn_map, all_imgs, k=10, ids=ids),
        "mAP@10": mean_average_precision_at_k(nn_map, all_imgs, k=10, ids=ids),
        "mAP@20": mean_average_precision_at_k(nn_map, all_imgs, k=20, ids=ids),
        "mAP@50": mean_average_precision_at_k(nn_map, all_imgs, k=50, ids=ids),
    }

    # Save ir metrics dict
    with open(out_file, 'w') as f:
        json.dump(ir_results, f, indent=2)


def main():
    for test in ["clip", "facenet_casia_webface", "facenet_vggface2"]:
        anonymized_path = Path(args.anonymized_dataset)
        find_neighbors(anonymized_path=f"{args.anonymized_dataset}",
                       original_path=f"{args.real_dataset}",
                       test=test,
                       k_sample=args.neighbors)
        compute_ir_metrics(path=Path(f'./') / anonymized_path.name,
                           test=f"{test}-emb",
                           files_path=Path(args.real_dataset))


if __name__ == '__main__':
    main()
