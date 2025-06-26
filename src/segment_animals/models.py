from typing import Annotated, Tuple
from pydantic import BaseModel


class AnimalDetection(BaseModel):
    """
    Model representing an animal detection.
    """

    bbox: Annotated[
        Tuple[float, float, float, float],
        "Bounding box coordinates (x_min, y_min, x_max, y_max)",
    ]
    confidence: Annotated[float, "Confidence score of the detection (0.0 to 1.0)"]


class AnimalSegment(BaseModel):
    """
    Model representing an animal segmentation.
    """

    mask: Annotated[
        list[list[float]],
        "Segmentation mask as a binary array",
    ]
