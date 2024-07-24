from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_root_path: Path,
        data_frame: pd.DataFrame,
        image_transform: Optional[torch.nn.Module] = None,
        mask_transform: Optional[torch.nn.Module] = None,
    ):
        """
        Args:
            data_root_path (Path): Root directory path for the dataset.
            data_frame (pd.DataFrame): DataFrame with volume, image paths, and segmentation paths.
            image_transform (Optional[torch.nn.Module], optional): Optional transform to be applied on an image.
            mask_transform (Optional[torch.nn.Module], optional): Optional transform to be applied on a mask.
        """
        self.data_root_path = data_root_path
        self.data_frame = data_frame
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_root_path / self.data_frame.iloc[idx, 1]
        segment_path = self.data_root_path / self.data_frame.iloc[idx, 2]

        img = torch.tensor(nib.load(img_path).get_fdata(), dtype=torch.float16)
        segment = torch.tensor(nib.load(segment_path).get_fdata(), dtype=torch.float16)

        if self.image_transform:
            img = self.image_transform(img)

        if self.mask_transform:
            segment = self.mask_transform(segment)

        return img.unsqueeze(0), segment.unsqueeze(0)
