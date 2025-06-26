from typing import List, Literal
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import torch
from segment_animals.models import AnimalDetection
from segment_animals.model_cache import get_model_path
from segment_anything.utils.transforms import ResizeLongestSide

SegmentationModelNames = Literal["vit_h", "vit_l", "vit_b"]


class SegmentationModel:
    """
    Model for segmenting animals in images using SAM (Segment Anything Model).
    """

    def __init__(
        self,
        model_name: SegmentationModelNames = "vit_h",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        # Get the model path, downloading it if necessary
        model_path = get_model_path(model_name, auto_download=True)

        self.sam = sam_model_registry[model_name](checkpoint=str(model_path))
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)
        self.resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)

    def segment(self, image: Image.Image, detections: List[AnimalDetection]):
        """
        Segment animals in the provided image.

        :param image: The input image to process.
        :param detections: The detections to guide the segmentation.
        :return: A list of segmentation masks.
        """
        np_image = np.array(image.convert("RGB"))
        self.predictor.set_image(np_image)

        if not detections:
            return torch.empty((0, *np_image.shape[:2]), dtype=torch.bool)

        transformed_boxes = self.resize_transform.apply_boxes_torch(
            torch.tensor(
                np.atleast_2d(
                    np.array(
                        [
                            [
                                d.bbox[0],
                                d.bbox[1],
                                d.bbox[0] + d.bbox[2],
                                d.bbox[1] + d.bbox[3],
                            ]
                            for d in detections
                        ]
                    )
                )
            ),
            np_image.shape[:2],
        ).to(self.sam.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return masks.cpu()
