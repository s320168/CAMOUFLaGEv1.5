from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_path,transform):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms

        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        folder_path = Path(folder_path)
        self.image_list = list(folder_path.rglob('*.*'))
        # Calculate len
        self.data_len = len(self.image_list)
        self.tfms = transform

    def __getitem__(self, index):
        single_image_path = self.image_list[index]
        im_as_im = Image.open(single_image_path).convert("RGB")
        return self.tfms(im_as_im), single_image_path.with_suffix("").name

    def __len__(self):
        return self.data_len