import os
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from generic_dataset import DatasetImgFolder
from ir_metrics import mean_average_precision_at_k, mean_recall_at_k
from tqdm import tqdm
import clip
import json
import argparse


parser = argparse.ArgumentParser(description="Image-level anonymization evaluation.")
parser.add_argument('--real_dataset_root', help="root directory of the real dataset", type=str, required=True)
parser.add_argument('--real_dataset_img_subfolder', help="img subfolder of the real dataset", type=str, required=True)
parser.add_argument('--anonymized_dataset_root', help="root directory of the real dataset", type=str, required=True)
parser.add_argument('--anonymized_dataset_img_subfolder', help="img subfolder of the real dataset", type=str, required=True)
parser.add_argument('--neighbors', help="number of neighbors to be retrieved", type=int, default=50)

args = parser.parse_args()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_type = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'
torch.set_default_tensor_type(tensor_type)


def extract_embeddings(src_dataset, subfolder):
    # Check if the dataset image folder exists
    if not os.path.isdir(src_dataset):
        raise FileNotFoundError(f'The directory "{src_dataset}/{subfolder}" does not exist.')

    embedding_dir = f'{src_dataset}/{subfolder}_embeddings'
    # Create the output dir
    os.makedirs(embedding_dir, exist_ok=True)

    # Check if embeddings already exists
    if os.path.isfile(f'{embedding_dir}/clip_embeds.pt'):
        print(f'Embeddings already available in {embedding_dir}. Consider to delete this file to get updated results.')
        return

    # === CLIP Image - embeds === #

    # Load CLIP model
    clip_model, clip_img_transform = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.float()
    clip_model.eval()

    # Create dataset and dataloader from original dataset folder
    dataset = DatasetImgFolder(root=f'{src_dataset}/{subfolder}', transform=clip_img_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    # Filenames and embeddings
    img_filenames = []
    clip_embeds = []

    for images, images_paths in tqdm(dataloader, desc="Computing image embeddings with CLIP: "):
        # Get image names
        img_filenames.extend([p.replace(f'{src_dataset}/{subfolder}/', '') for p in images_paths])

        # Extract embeddings
        with torch.no_grad():
            images = images.to(device)
            img_clip_embeds = clip_model.encode_image(images)

        clip_embeds.append(img_clip_embeds.cpu())

    # Save images filename
    with open(f'{embedding_dir}/full_imgs_filenames.txt', 'w') as f:
        for name in img_filenames:
            f.write(f'{name}\n')

    # Save embeddings
    clip_embeds = torch.cat(clip_embeds)
    torch.save(clip_embeds, f'{embedding_dir}/clip_embeds.pt')


def find_neighbors(anonymized_embeds_path, original_embeds_path, anonymized_filenames_path, original_filenames_path, test, k):
    # Check if NN map file already exists
    if os.path.isfile(f'./output/{test}-map.json'):
        print(f'Map file already available in {test}-map.json. Consider to delete this file to get updated results.')
        return

    # Check if embeddings exists
    if not os.path.isfile(anonymized_embeds_path):
        raise FileNotFoundError(f'The file "{anonymized_embeds_path}" does not exist.')
    if not os.path.isfile(original_embeds_path):
        raise FileNotFoundError(f'The file "{original_embeds_path}" does not exist.')
    if not os.path.isfile(anonymized_filenames_path):
        raise FileNotFoundError(f'The file "{anonymized_filenames_path}" does not exist.')
    if not os.path.isfile(original_filenames_path):
        raise FileNotFoundError(f'The file "{original_filenames_path}" does not exist.')

    # Create the output dir
    os.makedirs('./output', exist_ok=True)

    # Load filenames
    with open(anonymized_filenames_path) as f:
        anonymized_filenames = [n.strip() for n in f.readlines()]
    with open(original_filenames_path) as f:
        original_filenames = [n.strip() for n in f.readlines()]

    # Load embeddings
    anonymized_embeds = torch.load(anonymized_embeds_path).numpy()
    original_embeds = torch.load(original_embeds_path).numpy()

    # Fit NearestNeighbors model on original embeddings
    nn_model = NearestNeighbors(n_neighbors=k).fit(original_embeds)

    # Find nearest embeddings
    distances, indices = nn_model.kneighbors(anonymized_embeds)

    # Build NN map dicts
    nn_map = dict()
    nn_map_dist = dict()
    for i, anonymized_filename in enumerate(anonymized_filenames):
        nn_map[anonymized_filename] = [original_filenames[int(idx)] for idx in indices[i]]
        nn_map_dist[anonymized_filename] = distances[i].tolist()

    # Save NN map dicts
    with open(f'./output/{test}-map.json', 'w') as f:
        json.dump(nn_map, f, indent=2)
    with open(f'./output/{test}-map-dist.json', 'w') as f:
        json.dump(nn_map_dist, f, indent=2)


def compute_ir_metrics(nn_map_path, original_filenames_path, test):
    # Check if NN map file exists
    if not os.path.isfile(nn_map_path):
        raise FileNotFoundError(f'The file "{nn_map_path}" does not exist.')

    # Create the output dir
    os.makedirs('./output', exist_ok=True)

    # Load NN map file
    with open(nn_map_path, "r") as f:
        nn_map = json.load(f)
    # Load filenames
    with open(original_filenames_path, "r") as f:
        all_imgs = f.read().split('\n')


    # Compute ir metrics
    ir_results = {
        "Recall@1": mean_recall_at_k(nn_map, all_imgs, k=1),
        "Recall@3": mean_recall_at_k(nn_map, all_imgs, k=3),
        "Recall@5": mean_recall_at_k(nn_map, all_imgs, k=5),
        "Recall@10": mean_recall_at_k(nn_map, all_imgs, k=10),
        "mAP@10": mean_average_precision_at_k(nn_map, all_imgs, k=10),
        "mAP@20": mean_average_precision_at_k(nn_map, all_imgs, k=20),
        "mAP@50": mean_average_precision_at_k(nn_map, all_imgs, k=50)
    }

    # Save ir metrics dict
    with open(f'./output/{test}-ir_results.json', 'w') as f:
        json.dump(ir_results, f, indent=2)


def main():

    # Extract embedding from real dataset
    extract_embeddings(src_dataset=args.real_dataset_root,
                       subfolder=args.real_dataset_img_subfolder)
    # Extract embedding from anonymized dataset
    extract_embeddings(src_dataset=args.anonymized_dataset_root,
                       subfolder=args.anonymized_dataset_img_subfolder)

    # Find neighbors using CLIP embeddings
    find_neighbors(anonymized_embeds_path=f"{args.anonymized_dataset_root}/{args.anonymized_dataset_img_subfolder}_embeddings/clip_embeds.pt",
                   original_embeds_path=f"{args.real_dataset_root}/{args.real_dataset_img_subfolder}_embeddings/clip_embeds.pt",
                   anonymized_filenames_path=f"{args.anonymized_dataset_root}/{args.anonymized_dataset_img_subfolder}_embeddings/full_imgs_filenames.txt",
                   original_filenames_path=f"{args.real_dataset_root}/{args.real_dataset_img_subfolder}_embeddings/full_imgs_filenames.txt",
                   test='clip',
                   k=args.neighbors)

    # Compute ir metrics (Recall@K and mAP)
    compute_ir_metrics(nn_map_path="./output/clip-map.json",
                       original_filenames_path=f"{args.real_dataset_root}/{args.real_dataset_img_subfolder}_embeddings/full_imgs_filenames.txt",
                       test='clip')


if __name__ == '__main__':
    main()
