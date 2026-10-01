from typing import Literal, Sequence
import warnings
from PIL import Image
import torch
from .detect import DetectionModel, DetectionModelNames
from .segment import SegmentationModel, SegmentationModelNames
from .models import Detection, DetectionClass


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



class AutoSegmenter:
    """Detect and segment selected classes: animal, human, and/or vehicle.

    By default, only animals are selected. Pass detection_classes=("human",) for humans
    alone, or detection_classes=("animal", "human") for both.
    """

    def __init__(
        self,
        detection_model_name: DetectionModelNames = "redwood",
        detection_threshold: float = 0.15,
        segmentation_model_name: SegmentationModelNames = "sam2.1_hiera_large",
        segmentation_device: Literal["cpu", "cuda", "mps"] = get_default_device(),
        *,
        detection_classes: Sequence[DetectionClass] = ("animal",),
    ):
        self.detector = DetectionModel(
            model_name=detection_model_name,
            threshold=detection_threshold,
            detection_classes=detection_classes,
        )
        self.segmentor = SegmentationModel(
            model_name=segmentation_model_name, device=segmentation_device
        )

    def process_image(self, image: Image.Image) -> tuple:

        detections = self.detector.detect(image)
        masks = self.segmentor.segment(image, detections)
        return detections, masks


class AutoAnimalSegmenter(AutoSegmenter):
    """Deprecated compatibility wrapper; use AutoSegmenter instead."""

    def __init__(
        self,
        detection_model_name: DetectionModelNames = "redwood",
        detection_threshold: float = 0.15,
        segmentation_model_name: SegmentationModelNames = "sam2.1_hiera_large",
        segmentation_device: Literal["cpu", "cuda", "mps"] = get_default_device(),
        include_humans: bool = False,
    ):
        warnings.warn(
            "AutoAnimalSegmenter is deprecated; use AutoSegmenter() for animals "
            'or AutoSegmenter(detection_classes=("animal", "human")) to include humans.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            detection_model_name=detection_model_name,
            detection_threshold=detection_threshold,
            segmentation_model_name=segmentation_model_name,
            segmentation_device=segmentation_device,
            detection_classes=("animal", "human") if include_humans else ("animal",),
        )


def main() -> None:
    print(
        "No main function implemented yet. Use AutoSegmenter to process images."
    )
