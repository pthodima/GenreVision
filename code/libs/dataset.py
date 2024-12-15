import os
import random

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms

# from .transform import Compose, ToTensor


def trivial_batch_collator(batch):
    """
    A batch collator that allows us to bypass auto batching
    """
    return tuple(zip(*batch))

def worker_init_reset_seed(worker_id):
    """
    Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def build_dataset(name, split, img_folder, ann_folder):
    """
    Create VOC dataset with default transforms for training / inference.
    New datasets can be linked here.
    """
    if name == "MovieLens20M":
        assert split in ["train", "test", 'movie']
        is_training = split in ["train", "movie"]
    else:
        print("Unsupported dataset")
        return None

    if is_training: # Training
        tf = transforms.Compose([
            transforms.Resize(256),               # Resize keeping aspect ratio
            transforms.CenterCrop(224),           # Center crop to 224x224
            transforms.ToTensor(),                # Convert to tensor
            transforms.Normalize(                 # Normalize based on ImageNet mean and std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else: # Evaluation
        tf = transforms.Compose([
            transforms.Resize(256),               # Resize keeping aspect ratio
            transforms.CenterCrop(224),           # Center crop to 224x224
            transforms.ToTensor(),                # Convert to tensor
            transforms.Normalize(                 # Normalize based on ImageNet mean and std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    if name == "MovieLens20M":
        dataset = MovieLens20M(
            os.path.join(ann_folder, split + ".csv"),
            img_folder,
            tf
        )
    return dataset

def build_dataloader(dataset, is_training, batch_size, num_workers):
    """

    :param dataset: torch.utils.data.Dataset
    :param is_training: boolean
    :param batch_size: int
    :param num_workers: int
    :return: torch.utils.data.DataLoader for the dataset
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        persistent_workers=True,
    )
    return loader


class MovieLens20M(Dataset):
    def __init__(self, annotations_file, img_dir, transforms):
        self.img_dir = img_dir
        self._transforms = transforms
        self.data = pd.read_csv(annotations_file)

        # remove rows with no associated genre
        self.data = self.data[self.data['genres'] != "(no genres listed)"]

        # remove rows without posters by checking if the image file exists
        self.data = self.data[self.data['movieId'].apply(
            lambda x: os.path.exists(os.path.join(self.img_dir, str(x)+".jpg"))
        )]

        # split the genres into list
        self.data['genres'] = self.data['genres'].apply(lambda x: x.split('|'))
        # one-hot encode the genres
        self.mlb = MultiLabelBinarizer()
        self.genre_encoded = self.mlb.fit_transform(self.data['genres'])

        # store the one-hot encoded labels as a tensor
        self.genres = torch.tensor(self.genre_encoded, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.data.iloc[idx, 0]) + ".jpg")
        img = read_image(img_path)
        genres = self.data.iloc[idx, 1]
        if self._transforms:
            img = self._transforms(img)
        return img, genres

# if __name__ == "__main__":
#
#     dataset = build_dataset('MovieLens20M', 'movie', 'data/MovieLens20M/posters', 'data/MovieLens20M/')
#     dataloader = build_dataloader(dataset, True, 32, 1)
#
#     # print(dataset.mlb.classes_)
#
#     for i, (titles, labels) in enumerate(dataloader):
#         print(i)