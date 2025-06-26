# Segment Animals

Segment Animals is a Python package for segmenting (extracting) animals from images using deep learning models. It provides a pipeline that combines object detection and segmentation to identify and extract animals from images, making it useful for wildlife research, conservation efforts, and any application where you wish to remove the background from images containing animals.

Segment Animals builds upon the [Segment Anything](https://github.com/facebookresearch/segment-anything) and [MegaDetector](https://github.com/agentmorris/MegaDetector/blob/main/getting-started.md) models.

# Installation

You can install Segment Animals using pip:

```bash
pip install segment-animals
```

## Usage

Here's a quick example of how to use Segment Animals, for a more detailed guide refer to the [notebook](./notebook.ipynb).

### Importing the library and processing an image

```python
from segment_animals import AutoAnimalSegmenter
from segment_animals.util import load_image

model = AutoAnimalSegmenter()

image = load_image("path/to/your/image.jpg")

detections, masks = model.process_image(image)
print(f"Found {len(detections)} animals.")
```

### Visualizing detections and masks

```python
from segment_animals.viz import plot_detections_and_masks

plot_detections_and_masks(image, detections, masks)
```

You should then see a visualisation along the lines of this ([original image from Wikipedia](https://commons.wikimedia.org/wiki/File:Camouflaged_Predator.jpg))...

![Example Segmentation](./example_viz.png)

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

![Example Mask](./example_extract.png)

# Working with Segment Animals?

It'd be great to hear how you're using Segment Animals! Drop me a line at Benjamin.Evans at ioz.ac.uk or open an issue on the [GitHub repository](https://github.com/bencevans/segment-animals/issues).