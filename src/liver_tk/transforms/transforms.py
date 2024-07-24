from typing import Tuple
import torch


class PadOrTrim:
    """
    A torch transform that Adjusts the depth of the input image and segment to match the target
    depth by either padding or trimming.

    Args:
        image (torch.Tensor): The input image tensor.
        segment (torch.Tensor): The input segmentation tensor.
        target_depth (int): The target depth to which the image and segment should be adjusted.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The adjusted image and segmentation tensors.
    """

    def __init__(
        self,
        target_depth: int,
        sparse_result: bool = False,
        depth_dim: int = 2,
        pad: float = 0,
    ):
        self.target_depth = target_depth
        self.depth_dim = depth_dim
        self.pad = pad
        self.sparse_result = sparse_result

    def __call__(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current_depth = sample.shape[self.depth_dim]

        if current_depth > self.target_depth:
            sample = sample[:, :, : self.target_depth]
        elif current_depth < self.target_depth:
            pad_after = self.target_depth - current_depth
            sample = torch.nn.functional.pad(sample, (self.pad, pad_after))

        return sample


class WindowImage:
    """
    A torch transform tha applies Hounsfiled windowing to a CT image for better visualization.

    Args:
        image (torch.Tensor): The input CT image as a PyTorch tensor.
        window_level (float): The center of the windowing range.
        window_width (float): The width of the windowing range.

    Returns:
        torch.Tensor: The windowed image normalized to the range [0, 255] as a PyTorch tensor.
    """

    def __init__(self, window_level: float, window_width: float):
        self.window_level = window_level
        self.window_width = window_width

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        min_intensity = self.window_level - (self.window_width / 2)
        max_intensity = self.window_level + (self.window_width / 2)

        windowed_image = torch.clamp(image, min_intensity, max_intensity)
        windowed_image = (
            (windowed_image - min_intensity) / (max_intensity - min_intensity) * 255
        ).to(torch.float32)

        return windowed_image
