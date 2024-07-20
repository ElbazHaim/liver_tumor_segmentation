import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np


class NiftiDataset(Dataset):
    def __init__(self, csv_file: str, target_depth: int, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.target_depth = target_depth
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx, 1]
        seg_name = self.data_frame.iloc[idx, 2]

        image = nib.load(img_name).get_fdata()
        segment = nib.load(seg_name).get_fdata()

        image, segment = self.pad_or_trim(image, segment, self.target_depth)

        image = torch.tensor(image, dtype=torch.float16).unsqueeze(
            0
        )  # Adding channel dimension
        segment = torch.tensor(segment, dtype=torch.float16)

        if self.transform:
            image, segment = self.transform(image, segment)

        return image, segment

    def pad_or_trim(self, image, segment, target_depth):
        # Get current depth
        current_depth = image.shape[2]

        if current_depth > target_depth:
            start_idx = (current_depth - target_depth) // 2
            image = image[:, :, start_idx : start_idx + target_depth]
            segment = segment[:, :, start_idx : start_idx + target_depth]

        elif current_depth < target_depth:
            pad_before = (target_depth - current_depth) // 2
            pad_after = target_depth - current_depth - pad_before

            image = np.pad(image, ((0, 0), (0, 0), (pad_before, pad_after)), "constant")
            segment = np.pad(
                segment, ((0, 0), (0, 0), (pad_before, pad_after)), "constant"
            )

        return image, segment


if __name__ == "__main__":
    csv_file = "path/to/your/data.csv"
    target_depth = 64  # Example target depth
    dataset = NiftiDataset(csv_file=csv_file, target_depth=target_depth)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for batch in dataloader:
        images, segments = batch
        print(images.shape, segments.shape)
