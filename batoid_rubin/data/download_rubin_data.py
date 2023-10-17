from pathlib import Path
import requests
import zipfile
import io

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
    # Download the ZIP file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    if not args.outdir:
        outdir = Path(__file__).parent / args.dataset
    else:
        outdir = Path(args.outdir)

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
