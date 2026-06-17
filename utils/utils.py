import os
import zipfile
import requests


import os
import zipfile
import requests
from tqdm import tqdm


def download_and_extract(url, extract_dir="."):
    """Download a large zip archive from a Yandex Disk public link and extract it."""
    os.makedirs(extract_dir, exist_ok=True)

    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    response = requests.get(base_url + "public_key=" + url)
    response.raise_for_status()
    download_url = response.json()["href"]

    zip_path = os.path.join(extract_dir, "data.zip")

    # Stream download in chunks
    print("Downloading...")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk_size = 1024 * 1024 * 16

        with open(zip_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="data.zip",
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        with tqdm(total=len(members), desc="Extracting", unit="file") as bar:
            for member in members:
                zf.extract(member, extract_dir)
                bar.update(1)

    os.remove(zip_path)
    print("Done.")


def download(url, filename, save_dir="."):
    """
    Download a file from a Yandex Disk public link without extracting.

    Args:
        url: Yandex Disk public key (e.g. https://disk.yandex.ru/d/xxx).
        save_dir: Directory to save the file.
        filename: Name of the saved file.

    Returns:
        Path to the downloaded file.
    """
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    final_url = base_url + "public_key=" + url

    response = requests.get(final_url)
    download_url = response.json()["href"]
    download_response = requests.get(download_url)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        f.write(download_response.content)

    print(f"Downloaded to {save_path}")
    return save_path
