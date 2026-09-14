from typing import List, Literal

from PIL import Image
import torch
from transformers import Sam2Model, Sam2Processor

from segment_animals.models import AnimalDetection

SegmentationModelNames = Literal[
    "sam2.1_hiera_tiny",
    "sam2.1_hiera_small",
    "sam2.1_hiera_base_plus",
    "sam2.1_hiera_large",
]

MODEL_IDS = {
    "sam2.1_hiera_tiny": "facebook/sam2.1-hiera-tiny",
    "sam2.1_hiera_small": "facebook/sam2.1-hiera-small",
    "sam2.1_hiera_base_plus": "facebook/sam2.1-hiera-base-plus",
    "sam2.1_hiera_large": "facebook/sam2.1-hiera-large",
}


class SegmentationModel:
    """Model for segmenting animals using SAM 2.1 through Transformers."""

    def __init__(
        self,
        model_name: SegmentationModelNames = "sam2.1_hiera_large",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        if model_name not in MODEL_IDS:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {list(MODEL_IDS)}"
            )
        model_id = MODEL_IDS[model_name]
        self.sam = Sam2Model.from_pretrained(model_id).to(device).eval()
        self.processor = Sam2Processor.from_pretrained(model_id)

    @torch.inference_mode()
    def segment(self, image: Image.Image, detections: List[AnimalDetection]):
        """Return one boolean CPU mask per detection, shaped (N, 1, H, W).

        Detection boxes use pixel (x, y, width, height) coordinates. The
        processor resizes the XYXY prompts and restores masks to image size.
        The empty result retains the existing (0, H, W) shape.
        """
        if not detections:
            return torch.empty((0, image.height, image.width), dtype=torch.bool)

        boxes = [
            [x, y, x + width, y + height]
            for detection in detections
            for x, y, width, height in [detection.bbox]
        ]
        inputs = self.processor(
            images=image.convert("RGB"), input_boxes=[boxes], return_tensors="pt"
        ).to(self.sam.device)
        outputs = self.sam(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), binarize=True
        )[0]
        return masks.to(device="cpu", dtype=torch.bool)
