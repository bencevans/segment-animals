import torch
from megadetector.detection import run_detector
from .models import Detection, DetectionClass
from typing import List, Literal, Sequence
from logging import getLogger

logger = getLogger(__name__)

DetectionModelNames = Literal["MDV5A", "MDV5B", "redwood"]
CATEGORY_NAMES = {"1": "animal", "2": "human", "3": "vehicle"}


def _validate_classes(
    detection_classes: Sequence[DetectionClass],
) -> tuple[DetectionClass, ...]:
    if isinstance(detection_classes, str) or not detection_classes:
        raise ValueError(
            "detection_classes must be a non-empty sequence of animal, human, or vehicle"
        )
    selected = tuple(detection_classes)
    if any(name not in CATEGORY_NAMES.values() for name in selected):
        raise ValueError("Supported classes are animal, human, and vehicle")
    return selected


def _load_detector(model_name: str):
    """Load a MegaDetector model, including older YOLOv5 checkpoints.

    MegaDetector has the same compatibility fallback internally, but some Python
    versions report the missing class as ``module ... has no attribute`` instead
    of ``Can't get attribute``.  The latter is currently the only wording its
    fallback recognizes.
    """
    try:
        return run_detector.load_detector(model_name)
    except AttributeError as error:
        if (
            getattr(error, "name", None) != "DetectionModel"
            or getattr(getattr(error, "obj", None), "__name__", None)
            != "models.yolo"
        ):
            raise

        from models import yolo  # type: ignore[import-not-found]

        if not hasattr(yolo, "Model"):
            raise

        logger.info("Applying YOLOv5 DetectionModel compatibility alias")
        yolo.DetectionModel = yolo.Model
        return run_detector.load_detector(model_name)


class DetectionModel:
    """
    Model for detecting selected MegaDetector classes in images.
    """

    def __init__(
        self,
        model_name: str = "MDV5A",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        threshold: float = 0.15,
        include_humans: bool = False,
        *,
        detection_classes: Sequence[DetectionClass] | None = None,
    ):
        """
        Initialize the detection model with a specified model name.
        """
        if detection_classes is not None and include_humans:
            raise ValueError("Specify detection_classes or include_humans, not both")
        if detection_classes is None:
            detection_classes = ("animal", "human") if include_humans else ("animal",)
        selected = _validate_classes(detection_classes)
        self.categories = {
            key for key, name in CATEGORY_NAMES.items() if name in selected
        }
        self.device = torch.device(device)
        logger.info(f"Loading detection model '{model_name}' on device: {self.device}")

        self.model = _load_detector(model_name)
        logger.info(f"Model loaded: {self.model}")

        self.threshold = threshold

    def detect(self, image) -> List[Detection]:
        """
        Detect the selected classes in the provided image.

        :param image: The input image to process.
        :return: A list of detections above a confidence threshold.
        """
        result = self.model.generate_detections_one_image(
            image, image_id="", detection_threshold=self.threshold
        )

        return [
            Detection(
                bbox=(
                    d["bbox"][0] * image.width,
                    d["bbox"][1] * image.height,
                    d["bbox"][2] * image.width,
                    d["bbox"][3] * image.height,
                ),
                confidence=d["conf"],
                category=CATEGORY_NAMES[d["category"]],
            )
            for d in result["detections"]
            if d["conf"] >= self.threshold and d["category"] in self.categories
        ]
