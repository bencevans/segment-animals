"""
Model caching and download utilities for segment_animals.

Handles downloading and caching of model weights with hash verification.
"""

import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, List


# Model configurations with their download URLs, expected SHA256 hashes, and namespaces
MODEL_CONFIGS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "sha256": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
        "namespace": "sam",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "sha256": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
        "namespace": "sam",
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        "namespace": "sam",
    },
}


def get_cache_dir(model_namespace: str = "sam") -> Path:
    """
    Get the cache directory for a specific model namespace within segment_animals.

    Args:
        model_namespace: The namespace for the models (e.g., 'sam', 'yolo', etc.)

    Returns:
        Path to the cache directory for the specified namespace
    """
    if os.name == "nt":  # Windows
        cache_base = os.environ.get(
            "LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local")
        )
    else:  # macOS and Linux
        cache_base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))

    cache_dir = Path(cache_base) / "segment_animals" / "models" / model_namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """
    Verify that a file matches the expected SHA256 hash.

    Args:
        file_path: Path to the file to verify
        expected_hash: Expected SHA256 hash

    Returns:
        True if hash matches, False otherwise
    """
    if not file_path.exists():
        return False

    actual_hash = calculate_file_hash(file_path)
    return actual_hash.lower() == expected_hash.lower()


def download_file_with_progress(url: str, destination: Path) -> None:
    """
    Download a file with progress indication.

    Args:
        url: URL to download from
        destination: Path where to save the file
    """
    print(f"Downloading from {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)

                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end="", flush=True)

    if total_size > 0:
        print()  # New line after progress
    print(f"Download completed: {destination}")


def get_model_path(model_name: str, auto_download: bool = True) -> Path:
    """
    Get the path to a cached model, downloading it if necessary.

    Args:
        model_name: Name of the model (e.g., 'sam_vit_h_4b8939')
        auto_download: Whether to automatically download the model if not found

    Returns:
        Path to the model file

    Raises:
        ValueError: If model_name is not recognized
        FileNotFoundError: If model is not cached and auto_download is False
        RuntimeError: If download fails or hash verification fails
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]
    model_namespace = config.get(
        "namespace", "sam"
    )  # Default to 'sam' for backward compatibility
    cache_dir = get_cache_dir(model_namespace)
    model_path = cache_dir / config["filename"]

    # Check if model exists and has correct hash
    if model_path.exists():
        print(f"Found cached model: {model_path}")
        if verify_file_hash(model_path, config["sha256"]):
            print("Hash verification passed")
            return model_path
        else:
            print("Hash verification failed, re-downloading...")
            model_path.unlink()  # Remove corrupted file

    if not auto_download:
        raise FileNotFoundError(
            f"Model {model_name} not found in cache and auto_download is disabled"
        )

    # Download the model
    try:
        print(f"Downloading {model_name} model...")
        download_file_with_progress(config["url"], model_path)

        # Verify hash after download
        if not verify_file_hash(model_path, config["sha256"]):
            model_path.unlink()  # Remove corrupted download
            raise RuntimeError(
                f"Downloaded file hash verification failed for {model_name}"
            )

        print("Hash verification passed")
        return model_path

    except Exception as e:
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Failed to download {model_name}: {e}")


def register_model(
    model_name: str, url: str, filename: str, sha256: str, namespace: str = "sam"
) -> None:
    """
    Register a new model in the model registry.

    Args:
        model_name: Unique identifier for the model
        url: Download URL for the model
        filename: Local filename to save the model as
        sha256: Expected SHA256 hash of the model file
        namespace: Model namespace (default: 'sam')
    """
    MODEL_CONFIGS[model_name] = {
        "url": url,
        "filename": filename,
        "sha256": sha256,
        "namespace": namespace,
    }


def list_models(namespace: Optional[str] = None) -> List[str]:
    """
    List available models, optionally filtered by namespace.

    Args:
        namespace: Optional namespace to filter by

    Returns:
        List of model names
    """
    if namespace is None:
        return list(MODEL_CONFIGS.keys())

    return [
        model_name
        for model_name, config in MODEL_CONFIGS.items()
        if config.get("namespace", "sam") == namespace
    ]


def list_namespaces() -> List[str]:
    """
    List all available model namespaces.

    Returns:
        List of unique namespaces
    """
    namespaces = set()
    for config in MODEL_CONFIGS.values():
        namespaces.add(config.get("namespace", "sam"))
    return sorted(list(namespaces))


def get_model_info(model_name: str) -> Dict[str, str]:
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_name].copy()
