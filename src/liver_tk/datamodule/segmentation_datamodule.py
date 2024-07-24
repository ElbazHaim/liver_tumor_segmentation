from pathlib import Path

import torch
import lightning as L
import pandas as pd
from liver_tk.datamodule.segmentation_dataset import SegmentationDataset
from liver_tk.transforms import PadOrTrim, WindowImage
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root_path: str,
        csv_file_path: str,
        batch_size: int,
        num_workers: int = 4,
        window_level: int = 30,
        window_width: int = 150,
        target_depth: int = 852,
    ):
        super().__init__()
        self.data_root_path = Path(data_root_path)
        self.csv_file_path = Path(csv_file_path)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.df_train: pd.DataFrame = None
        self.df_val: pd.DataFrame = None
        self.df_test: pd.DataFrame = None

        self.image_transform = transforms.Compose(
            [
                WindowImage(window_level=window_level, window_width=window_width),
                PadOrTrim(target_depth=target_depth),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: (x > 0).to(torch.float16)),
                PadOrTrim(target_depth=target_depth),
            ]
        )

    def setup(self, stage: str = None):
        train_size: int = 0.7
        val_size: int = 0.15
        test_size: int = 0.15
        df = pd.read_csv(self.csv_file_path)
        self.df_train, df_temp = train_test_split(
            df, test_size=(1.0 - train_size), random_state=42
        )
        test_frac = test_size / (val_size + test_size)
        self.df_val, self.df_test = train_test_split(
            df_temp, test_size=test_frac, random_state=42
        )

        self.train_set = SegmentationDataset(
            data_root_path=self.data_root_path,
            data_frame=self.df_train,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
        )

        self.val_set = SegmentationDataset(
            data_root_path=self.data_root_path,
            data_frame=self.df_val,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
        )

        self.test_set = SegmentationDataset(
            data_root_path=self.data_root_path,
            data_frame=self.df_test,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
