# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from typing import Literal, Union, List
from segment_animals import AutoAnimalSegmenter
from segment_animals.util import load_image
from segment_animals.viz import plot_detections_and_masks, extract_masks


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoAnimalSegmenter()

    def predict(
        self,
        image: Path = Input(description="Camera trap image to run prediction on"),
        format: Literal["json", "image", "extracts"] = Input(
            description="Output format", default="image"
        ),
    ) -> Union[Path, List[Path], List[dict]]:
        """Run a single prediction on the model"""
        image = load_image(image)

        detections, masks = self.model.process_image(image)

        if format == "json":
            return [
                {
                    "bbox": detection.bbox,
                    "confidence": detection.confidence,
                    "class": detection.class_name,
                    "mask": mask.tolist(),
                }
                for detection, mask in zip(detections, masks)
            ]
        elif format == "image":
            output_image = plot_detections_and_masks(image, detections, masks)
            output_path = Path("output.png")
            output_image.savefig(output_path)
            return output_path
        elif format == "extracts":
            for i, mask_extract in enumerate(
                extract_masks(image, masks, whole_image=False)
            ):
                mask_extract.save(f"animal_mask_{i}.png")

            return [Path(f"animal_mask_{i}.png") for i in range(len(masks))]
