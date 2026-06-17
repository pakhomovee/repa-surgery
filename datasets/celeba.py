import os
import io
import zipfile
from collections import OrderedDict

import torch
import gdown
from torch.utils.data import Dataset
from torchvision import transforms
import re
from PIL import Image

# Dataset was already provided
try:
    from natsort import natsorted
    SORTFN = natsorted
except ImportError:
    # fallback to regular sorted if natsort is not available
    SORTFN = sorted


class CelebADataset(Dataset):
    """
    Custom Dataset class for the CelebA dataset.
    Automatically downloads data and annotations from Google Drive
    if not found in the specified root_dir.

    Args:
        root_dir (str): The directory to store or locate the CelebA data.
        transform (callable, optional): Optional transform to be applied on a PIL Image sample.
    """

    def __init__(self, root_dir: str = "../data/celeba",
                 transform: transforms.Compose = None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.header = None  # will store attribute names

        self.dataset_folder = os.path.join(root_dir, "img_align_celeba")
        if not os.path.isdir(self.dataset_folder):
            # The unzipped folder will be root_dir/img_align_celeba
            self.download_images()

        # Load file names - keep only files, not subdirectories.
        # gdown sometimes extracts into a nested subdirectory
        # (e.g. img_align_celeba/celeba/), so we filter those out here.
        all_entries = os.listdir(self.dataset_folder)
        self.filenames = [f for f in all_entries
                          if os.path.isfile(os.path.join(self.dataset_folder, f))]
        # Ensure consistent ordering (important if you rely on index-based consistency)
        self.filenames = SORTFN(self.filenames)

        attr_folder = os.path.join(root_dir, "annotations")
        if not os.path.isdir(attr_folder):
            os.makedirs(attr_folder, exist_ok=True)
            self.download_annotations(attr_folder)

        # Check for attribute files in the provided folder
        attr_file_path = os.path.join(attr_folder, "list_attr_celeba.txt")
        if not os.path.isfile(attr_file_path):
            raise FileNotFoundError(
                f"Could not find list_attr_celeba.txt in the annotations folder."
            )

        # Locate and parse the list_attr_celeba.txt file
        self.annotations = []
        with open(attr_file_path, 'r') as f:
            lines = f.read().splitlines()

        # Load attributes
        for i, line in enumerate(lines):
            # The rest lines each correspond to one image
            line = re.sub(r'\s+', ' ', line.strip())
            if i == 0:
                continue  # number of images
            elif i == 1:
                # line might have variable spaces, so split robustly
                self.header = line.split()  # header line with attribute names
            else:
                parts = line.split()
                filename = parts[0]
                attr_vals = [int(val) for val in parts[1:]]
                self.annotations.append((filename, attr_vals))

        # the rest are attribute labels
        self.attr_map = {
            fn: torch.tensor(attr_vals, dtype=torch.long)
            for fn, attr_vals in self.annotations
        }

    def download_images(self):
        """If the CelebA folder isn't found, download and extract the image dataset from Google Drive."""
        os.makedirs(self.root_dir, exist_ok=True)
        zip_path = os.path.join(self.root_dir, "img_align_celeba.zip")
        if not os.path.isfile(zip_path):
            file_id = "1zVyBr0Q667RK_0j6QANLAajNQxp3yWOl"
            gdown.download(id=file_id, output=zip_path, quiet=False)
        else:
            print(f"Found existing ZIP file at {zip_path}. Skipping download.")
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(self.root_dir)
        print("Extraction finished.")

    def download_annotations(self, annotation_folder):
        """Download annotations from Google Drive if they are missing."""
        attr_file_path = os.path.join(annotation_folder, "list_attr_celeba.txt")
        if not os.path.isfile(attr_file_path):
            print("Downloading annotations for CelebA...")
            annotation_url = "https://drive.google.com/drive/folders/1tUfjh4c7ss8Bb-w5Fa5l35Ei2ep86yNd"
            gdown.download_folder(url=annotation_url, output=annotation_folder, quiet=False, use_cookies=False)
        else:
            print(f"Annotations already exist in {annotation_folder}. Skipping download.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of the sample
        Returns:
            tuple: (image, info_dict) where
                image is the transformed PIL image,
                info_dict is a dictionary containing:
                    - 'filename' (str)
                    - 'idx' (int)
                    - 'attributes' (torch.Tensor of shape [attributes])
        """
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        # Load the image
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # Fetch attributes if they exist in the attr_map (some extra files might appear)
        attributes = self.attr_map.get(img_name, torch.zeros(40, dtype=torch.long))
        info_dict = {
            'filename': img_name,
            'idx': idx,
            'attributes': attributes
        }
        return img, info_dict


def make_get_class(header: list[str], selected_attrs: list[str]):
    """
    Return domain mask by info_dict indices.
    """
    indices = [header.index(a) for a in selected_attrs]
    powers = 2 ** torch.arange(len(indices))

    def get_class_fn(info_dict: dict) -> torch.Tensor:
        bits = info_dict['attributes'][:, indices].clamp(min=0)
        return (bits * powers).sum(dim=1).long()

    return get_class_fn  # Return both
