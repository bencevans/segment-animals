from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import random


def thumbnail(image, size=512):
    """
    Create a thumbnail of the image with the specified size.
    The thumbnail will maintain the aspect ratio of the original image.
    """
    image = image.copy()
    image.thumbnail((size, size))
    return image


def random_bright_rgb_0_1():
    """
    Generate a random bright RGB color with values between 0 and 1.
    The color will be bright enough to be visible against most backgrounds.
    """
    return tuple(random.randint(128, 255) / 255 for _ in range(3))


def plot_detections_and_masks(image, detections, masks):
    """
    Plot the detections and masks on the image.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for i, (detection, mask) in enumerate(zip(detections, masks)):
        random_color = random_bright_rgb_0_1()

        plt.gca().add_patch(
            Rectangle(
                (detection.bbox[0], detection.bbox[1]),
                detection.bbox[2],
                detection.bbox[3],
                fill=False,
                color=random_color,
            )
        )
        plt.text(
            detection.bbox[0] + 8,
            detection.bbox[1] - 20,
            f"Animal {i + 1}: {detection.confidence:.2f}",
            color="white",
            fontsize=12,
            bbox=dict(
                facecolor=random_color,
                alpha=0.5,
                edgecolor="none",
                boxstyle="round,pad=0.3",
            ),
        )

        # Create an overlay for the mask in the same color
        mask_2d = mask.squeeze()  # Remove extra dimension
        overlay = np.zeros_like(image, dtype=np.float32)
        overlay[mask_2d] = random_color
        plt.imshow(overlay, alpha=0.5)

    plt.axis("off")
    plt.show()
    return plt.gcf()  # Return the figure object for further manipulation if needed


def mask_to_bbox(mask):
    """Convert a mask to a bounding box."""

    mask_2d = mask.squeeze().numpy().astype("uint8")
    mask_pil = Image.fromarray(mask_2d * 255, mode="L")
    bbox = mask_pil.getbbox()
    return bbox


def extract_masks(image, masks, whole_image=False):
    """Extracts the original image filtered by each detected animal mask, with background transparent."""

    extracted_images = []
    image_np = np.array(image.convert("RGBA"))
    for mask in masks:
        mask_2d = mask.squeeze().numpy().astype("uint8")
        # Create an alpha channel based on the mask
        alpha = (mask_2d * 255).astype("uint8")
        # Stack alpha channel to image
        rgba = image_np.copy()
        rgba[..., 3] = alpha
        filtered_img = Image.fromarray(rgba, mode="RGBA")
        extracted_images.append(filtered_img)

        if not whole_image:
            # Crop the image to the bounding box of the mask
            bbox = mask_to_bbox(mask)
            filtered_img = filtered_img.crop(bbox)
            extracted_images[-1] = filtered_img

    return extracted_images
