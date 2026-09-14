# Segment Animals

Segment Animals is a Python package for segmenting (extracting) animals from images using deep learning models. It provides a pipeline that combines object detection and segmentation to identify and extract animals from images, making it useful for wildlife research, conservation efforts, and any application where you wish to remove the background from images containing animals.

Segment Animals builds upon the [Segment Anything 2 (SAM 2.1)](https://github.com/facebookresearch/sam2) and [MegaDetector](https://github.com/agentmorris/MegaDetector/blob/main/getting-started.md) models.

# Installation

Install Segment Animals and its dependencies with pip:

```bash
pip install segment-animals
```

## Usage

Here's a quick example of how to use Segment Animals, for a more detailed guide refer to the [notebook](https://github.com/bencevans/segment-animals/blob/main/notebook.ipynb).

### Importing the library and processing an image

```python
from segment_animals import AutoAnimalSegmenter
from segment_animals.util import load_image

model = AutoAnimalSegmenter()

image = load_image("path/to/your/image.jpg")

detections, masks = model.process_image(image)
print(f"Found {len(detections)} animals.")
```

### Choosing a segmentation model

The default is `sam2.1_hiera_large`. For a smaller model, pass
`segmentation_model_name` when creating the pipeline:

```python
model = AutoAnimalSegmenter(segmentation_model_name="sam2.1_hiera_tiny")
```

Available models are `sam2.1_hiera_tiny`, `sam2.1_hiera_small`,
`sam2.1_hiera_base_plus`, and `sam2.1_hiera_large`. Checkpoints are downloaded
on first use through Hugging Face Transformers and reused from the Hugging Face
cache. Set `HF_HOME` to choose a different cache location.
The former `vit_h`, `vit_l`, and `vit_b` names are no longer supported.

For development, install dependencies with `uv sync`.
CPU, CUDA, and MPS devices can be selected with `segmentation_device`.

### Visualizing detections and masks

```python
from segment_animals.viz import plot_detections_and_masks

plot_detections_and_masks(image, detections, masks)
```

You should then see a visualisation along the lines of this ([original image from Wikipedia](https://commons.wikimedia.org/wiki/File:Camouflaged_Predator.jpg))...

![Example Segmentation](https://raw.githubusercontent.com/bencevans/segment-animals/main/example_viz.png)

### Extracting and saving masks

```python
from segment_animals.viz import extract_masks

# Setting whole_image to False will return individual masks cropped to the extent
# of the predicted masks.
for i, mask_extract in enumerate(extract_masks(image, masks, whole_image=False)):
    # mask_extract is a PIL Image object so you can save it or manipulate it further
    mask_extract.save(f"animal_mask_{i}.png")
```

Resulting in something like this:

![Example Mask](https://raw.githubusercontent.com/bencevans/segment-animals/main/example_extract.png)

## Working with Segment Animals?

It'd be great to hear how you're using Segment Animals! Drop me a line at Benjamin.Evans at ioz.ac.uk or open an issue on the [GitHub repository](https://github.com/bencevans/segment-animals/issues).
