from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from annoy import AnnoyIndex

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from insightface.app import FaceAnalysis
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from dataset import CustomDatasetFromFile


class AnnoyWrapper:

    def __init__(self, method="buffalo_l", device='cpu'):
        self.device = device

        if method == "InceptionResnetV1":
            self.m = InceptionResnetV1(pretrained='vggface2').to(device).eval()
            self.model = lambda x, device: self.m(x.to(device))
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((160, 160)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # lambda x: x-0.5
            ])
            f = 512
        elif "CLIP" in method:
            method = method.split(":")[1]
            self.m = CLIPVisionModelWithProjection.from_pretrained(method).to(self.device, dtype=torch.float16).eval()
            self.model = lambda x, device: self.m(x.pixel_values.squeeze(1).to(device)).image_embeds
            self.transform = partial(CLIPImageProcessor(), return_tensors="pt")
            f = 512
        elif method == "buffalo_l":
            self.m = FaceAnalysis(name='buffalo_l')
            self.m.prepare(ctx_id=0, det_size=(128, 128))
            def func(x, device):
                embs = []
                for im, _ in x:
                    im = np.array(im)
                    im = im[:, :, ::-1]
                    im = self.m.get(im, max_num=1)
                    if len(im) == 0:
                        continue
                    im = im[0].normed_embedding
                    embs.append(torch.tensor(im))
                return torch.stack(embs)

            self.model = func
            self.transform = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((512, 512)),
                # np.array
            ])
            f = 512
        else:
            raise NotImplementedError()

        self.annoy = AnnoyIndex(f, 'angular')

    def clip_transforms(self, clip_t, images):
        return clip_t(images=images, return_tensors="pt").pixel_values

    def generate(self, folder, batch_size=512, workers=8, n_trees=10, name="test.ann"):
        dataset = CustomDatasetFromFile(folder, self.transform)
        print(f"Found {len(dataset)} images")
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
        i = 0
        for batch in tqdm(dataloader):
            images, _ = batch
            out = self.model(images, self.device).detach().cpu().numpy()
            for k in range(out.shape[0]):
                self.annoy.add_item(i, out[k])
                i += 1

        self.annoy.build(n_trees, workers)
        self.annoy.save(name)

    def get_nns_by_vector(self, file, n=1, search_k=-1, include_distances=False):
        if isinstance(file, (str, Path)):
            file = Image.open(file).convert("RGB")
        file = self.transform(file)
        if hasattr(file, "keys"):
            file.pixel_values = file.pixel_values.unsqueeze(0)
        else:
            file = file.unsqueeze(0)
        out = self.model(file, self.device).detach().cpu().numpy()
        return self.annoy.get_nns_by_vector(out[0], n, search_k, include_distances)
