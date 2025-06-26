from PIL import Image
from requests import get
from io import BytesIO
from pathlib import Path

ImageSource = str | Path | Image.Image


def load_image(source: ImageSource) -> Image.Image:
    """
    Load an image from a file path or URL.

    :param source: The file path or URL of the image.
    :return: The loaded image as a PIL Image object.
    """
    if isinstance(source, Image.Image):
        # If the source is already a PIL Image, return a copy of it
        return source.copy()

    elif isinstance(source, (Path, str)) and Path(source).is_file():
        # If the source is a file path, open the image from the file
        return Image.open(source)

    elif isinstance(source, str):
        # If the source is a URL, fetch the image from the URL
        response = get(source)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    else:
        raise ValueError(
            "Invalid image source. Provide a file path, URL, or PIL Image object."
        )
