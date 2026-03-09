from pathlib import Path
import requests
import zipfile
import io
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


zenodo_dois = {
    "bend_legacy": 8383823,
    "fea_legacy": 8384326,
    "bend": 8384775,
    "bend_ptt": 8384781,
}


def download_rubin_data(args):
    DOI = zenodo_dois.get(args.dataset, None)
    if DOI is None:
        raise ValueError(f"Unknown dataset {args.dataset}")

    api = r"https://zenodo.org/api/records/"
    url = f"{api}{DOI}/files-archive"

    if args.outdir is not None:
        outdir = Path(args.outdir)
    else:
        env_dir = os.environ.get("BATOID_RUBIN_DATA_DIR")
        if env_dir:
            outdir = Path(env_dir) / args.dataset
        else:
            outdir = Path(__file__).parent / args.dataset

    # Download the ZIP file with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()

    buffer = io.BytesIO()
    for data in tqdm(
        response.iter_content(chunk_size=8192),
        unit='KB',
        desc="Downloading"
    ):
        buffer.write(data)

    # Ensure output directory exists after successful download
    outdir.mkdir(parents=True, exist_ok=True)

    # Unzip the downloaded content
    with zipfile.ZipFile(buffer) as z:
        for member in z.infolist():
            z.extract(member, path=outdir)

    print(f"Downloaded and extracted ZIP file from {url} to {outdir}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=zenodo_dois.keys()
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None
    )
    args = parser.parse_args()
    download_rubin_data(args)
