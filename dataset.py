import os
from typing import Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

from .utils import HAM_CLASSES, resolve_image_path


class Ham10000Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, data_root: str, is_train: bool, image_size: int = 224) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.is_train = is_train
        self.image_size = image_size
        self._build_transforms()

    def _build_transforms(self) -> None:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.is_train:
            self.tf = T.Compose([
                T.Resize(int(self.image_size * 1.2)),
                T.RandomResizedCrop(self.image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(self.image_size * 1.15)),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label_name = row["dx"]
        label = HAM_CLASSES.index(label_name)
        image = Image.open(img_path).convert("RGB")
        tensor = self.tf(image)
        return tensor, label, os.path.basename(img_path)


def load_metadata_with_paths(data_root: str) -> pd.DataFrame:
    csv_path = os.path.join(data_root, "HAM10000_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns or "dx" not in df.columns:
        raise ValueError("Expected columns 'image_id' and 'dx' in metadata")

    filepaths = []
    for image_id in df["image_id"].astype(str).tolist():
        filepaths.append(resolve_image_path(data_root, image_id))
    df["filepath"] = filepaths

    df = df[df["dx"].isin(HAM_CLASSES)].dropna(subset=["filepath", "dx"]).reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42):
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["dx"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
