"""
Script to download DeepSub datasets from Zenodo.
"""

import os
import requests
from tqdm import tqdm

DATA_URLS = {
    "test.pt": "https://zenodo.org/records/15674391/files/test.pt",
    "train.pt": "https://zenodo.org/records/15674391/files/train.pt",
    "val.pt":   "https://zenodo.org/records/15674391/files/val.pt",
}


def download_file(url, dest_path):
    """
    Download a file from a URL to a local path with a progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=os.path.basename(dest_path),
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def main():
    """
    Create the datasets directory and download all dataset files.
    """
    os.makedirs("datasets", exist_ok=True)
    for filename, url in DATA_URLS.items():
        dest = os.path.join("datasets", filename)
        if os.path.exists(dest):
            print(f"{filename} already exists, skipping.")
        else:
            print(f"Downloading {filename}...")
            download_file(url, dest)
    print("All files downloaded.")


if __name__ == "__main__":
    main() 