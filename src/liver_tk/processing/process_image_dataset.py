import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

data_dir = Path("/home/haim/code/tumors/data/")
segment_dir = Path("/home/haim/code/tumors/data/segmentations/")


def set_depth(image, segment, target_depth):
    current_depth = image.shape[2]

    if current_depth > target_depth:
        start_idx = (current_depth - target_depth) // 2
        image = image[:, :, start_idx : start_idx + target_depth]
        segment = segment[:, :, start_idx : start_idx + target_depth]
    elif current_depth < target_depth:
        pad_before = (target_depth - current_depth) // 2
        pad_after = target_depth - current_depth - pad_before

        image = np.pad(image, ((0, 0), (0, 0), (pad_before, pad_after)), "constant")
        segment = np.pad(segment, ((0, 0), (0, 0), (pad_before, pad_after)), "constant")

    return image, segment


def hounsefield_window(image: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    """
    Apply windowing to a CT image for better visualization.

    Args:
        image (np.ndarray): The input CT image as a NumPy array.
        window_level (float): The center of the windowing range.
        window_width (float): The width of the windowing range.

    Returns:
        np.ndarray: The windowed image normalized to the range [0, 255] as a NumPy array.
    """
    min_intensity: float = window_level - (window_width / 2)
    max_intensity: float = window_level + (window_width / 2)
    
    windowed_image = np.clip(image, min_intensity, max_intensity)
    windowed_image = (
        (windowed_image - min_intensity) / (max_intensity - min_intensity) * 255
        ).astype(np.uint8)
    
    return windowed_image

def process_and_save_files(csv_file, output_image_dir, output_segment_dir, target_depth, window_level, window_width):
    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_segment_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        img_path = row['volume_path']
        seg_path = row['segment_path']

        img_nii = nib.load(img_path)
        seg_nii = nib.load(seg_path)

        image = img_nii.get_fdata()
        segment = seg_nii.get_fdata()

        affine_matrix = img_nii.affine

        processed_image, processed_segment = set_depth(image, segment, target_depth)

        windowed_image = hounsefield_window(processed_image, window_level, window_width)

        img_filename = os.path.basename(img_path)
        seg_filename = os.path.basename(seg_path)

        processed_img_path = os.path.join(output_image_dir, img_filename)
        processed_seg_path = os.path.join(output_segment_dir, seg_filename)

        nib.save(nib.Nifti1Image(windowed_image, affine_matrix), processed_img_path)
        nib.save(nib.Nifti1Image(processed_segment, affine_matrix), processed_seg_path)

        print(f"Processed and saved {img_filename} and {seg_filename}")


if __name__ == "__main__":
    csv_file = "/home/haim/code/tumors/liver_tumors/image_and_segment_paths.csv"
    output_image_dir = "/home/haim/code/tumors/data/processed/volumes"
    output_segment_dir = "/home/haim/code/tumors/data/processed/segmentations"
    target_depth = 851
    process_and_save_files(csv_file, output_image_dir, output_segment_dir, target_depth)
