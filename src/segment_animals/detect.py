import torch
from megadetector.detection import run_detector
from .models import AnimalDetection
from typing import List, Literal
from logging import getLogger

logger = getLogger(__name__)

DetectionModelNames = Literal["MDV5A", "MDV5B"]


class DetectionModel:
    """
    Model for detecting animals in images.
    """

    def __init__(
        self,
        model_name: str = "MDV5A",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        threshold: float = 0.15,
    ):
        """
        Initialize the detection model with a specified model name.
        """
        self.device = torch.device(device)
        logger.info(f"Loading detection model '{model_name}' on device: {self.device}")

        self.model = run_detector.load_detector(model_name)
        logger.info(f"Model loaded: {self.model}")

        self.threshold = threshold

    def detect(self, image) -> List[AnimalDetection]:
        """
        Detect animals in the provided image.

        :param image: The input image to process.
        :return: A list of detections above a confidence threshold.
        """
        result = self.model.generate_detections_one_image(
            image, image_id="", detection_threshold=self.threshold
        )

        return [
            AnimalDetection(
                bbox=(
                    d["bbox"][0] * image.width,
                    d["bbox"][1] * image.height,
                    d["bbox"][2] * image.width,
                    d["bbox"][3] * image.height,
                ),
                confidence=d["conf"],
            )
            for d in result["detections"]
            if d["conf"] >= self.threshold and d["category"] == "1"
        ]
