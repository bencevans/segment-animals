from typing import Annotated, Literal, Tuple
from pydantic import BaseModel


DetectionClass = Literal["animal", "human", "vehicle"]


class Detection(BaseModel):
    """
    Model representing a detection from MegaDetector.
    """

    bbox: Annotated[
        Tuple[float, float, float, float],
        "Bounding box coordinates (x_min, y_min, width, height)",
    ]
    confidence: Annotated[float, "Confidence score of the detection (0.0 to 1.0)"]
    category: DetectionClass = "animal"


# Preserve the original public name for existing callers.
AnimalDetection = Detection


class AnimalSegment(BaseModel):
    """
    Model representing an animal segmentation.
    """

    mask: Annotated[
        list[list[float]],
        "Segmentation mask as a binary array",
    ]
