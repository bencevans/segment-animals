from typing import Literal
from PIL import Image
import torch
from .detect import DetectionModel, DetectionModelNames
from .segment import SegmentationModel, SegmentationModelNames


def get_default_device() -> Literal["cpu", "cuda", "mps"]:
    """
    Get the default device for model inference.
    This function checks for CUDA availability first, then MPS (for Apple Silicon),
    and defaults to CPU if neither is available.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif "mps" in dir(torch.backends) and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class AutoAnimalSegmenter:
    """
    AutoAnimalSegmenter is a class that combines detection and segmentation models
    to automatically detect and segment animals in images.
    """

    def __init__(
        self,
        detection_model_name: DetectionModelNames = "MDV5A",
        detection_threshold: float = 0.15,
        segmentation_model_name: SegmentationModelNames = "default",
        segmentation_device: Literal["cpu", "cuda", "mps"] = get_default_device(),
    ):
        self.detector = DetectionModel(
            model_name=detection_model_name, threshold=detection_threshold
        )
        self.segmentor = SegmentationModel(
            model_name=segmentation_model_name, device=segmentation_device
        )

    def process_image(self, image: Image.Image) -> tuple:

        detections = self.detector.detect(image)
        masks = self.segmentor.segment(image, detections)
        return detections, masks


def main() -> None:
    print(
        "No main function implemented yet. Use AutoAnimalSegmenter to process images."
    )
