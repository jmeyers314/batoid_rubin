from pathlib import Path
import hashlib
import os
import requests
import shutil
import tempfile
import time

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
    "ccd_height_map": 18929776,
}


class ChecksumMismatch(Exception):
    """Raised when a downloaded file doesn't match its advertised checksum."""


def _umask():
    """Read the current umask without permanently changing it."""
    current = os.umask(0o022)
    os.umask(current)
    return current


def _retryable(exc):
    """Whether an exception is worth retrying.

    Client errors other than 429 won't fix themselves, so don't burn retries
    (or obscure the error message) on them.
    """
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        if response is None:
            return True
        return response.status_code == 429 or response.status_code >= 500
    return True


def _with_retries(func, retries=3, delay=0.5):
    """Call func, retrying transient failures."""
    for attempt in range(retries + 1):
        try:
            return func()
        except (requests.exceptions.RequestException, ChecksumMismatch) as e:
            if attempt == retries or not _retryable(e):
                raise
            print(f"  {type(e).__name__}: {e}; retrying ({attempt + 1}/{retries})")
            time.sleep(delay)


def _download_file(url, dest, size=None, checksum=None):
    """Download a single file, verifying its checksum if provided."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    md5 = hashlib.md5()
    with open(dest, "wb") as f:
        for data in tqdm(
            response.iter_content(chunk_size=8192),
            total=(size // 8192 + 1) if size else None,
            unit="KB",
            desc=f"Downloading {dest.name}",
            leave=False,
        ):
            md5.update(data)
            f.write(data)

    if checksum is not None:
        algo, _, expected = checksum.partition(":")
        if algo == "md5" and md5.hexdigest() != expected:
            raise ChecksumMismatch(
                f"{dest.name}: expected {expected}, got {md5.hexdigest()}"
            )


def download_rubin_data(args):
    DOI = zenodo_dois.get(args.dataset, None)
    if DOI is None:
        raise ValueError(f"Unknown dataset {args.dataset}")

    retries = getattr(args, "retries", 3)

    api = r"https://zenodo.org/api/records/"
    url = f"{api}{DOI}"

    if args.outdir is not None:
        outdir = Path(args.outdir)
    else:
        env_dir = os.environ.get("BATOID_RUBIN_DATA_DIR")
        if env_dir:
            outdir = Path(env_dir) / args.dataset
        else:
            outdir = Path(__file__).parent / args.dataset

    # Refuse to write into an existing directory: the atomic rename below
    # can't merge, and a populated outdir means someone else already
    # provided the data.
    if any(outdir.iterdir()) if outdir.is_dir() else outdir.exists():
        raise FileExistsError(
            f"Output directory {outdir} already exists and is not empty"
        )

    # Zenodo retired the /files-archive bulk-zip endpoint, so fetch the record
    # metadata and download each file individually.
    def get_metadata():
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["files"]

    files = _with_retries(get_metadata, retries=retries)

    # Download into a staging directory so a failure partway through doesn't
    # leave a half-populated outdir, which callers treat as already complete.
    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=outdir.parent, prefix=f".{outdir.name}-")
    )
    try:
        for finfo in files:
            _with_retries(
                lambda finfo=finfo: _download_file(
                    finfo["links"]["self"],
                    staging / finfo["key"],
                    size=finfo.get("size"),
                    checksum=finfo.get("checksum"),
                ),
                retries=retries,
            )

        # mkdtemp makes the staging dir 0700; the published data dir should
        # have normal umask-derived permissions so other users can read it.
        staging.chmod(0o777 & ~_umask())

        # Atomically publish the fully-downloaded directory. rename() replaces
        # an empty destination but fails on a populated one, which is what we
        # want: never merge into an existing dataset directory.
        staging.rename(outdir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Downloaded {len(files)} files from {url} to {outdir}")


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
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of times to retry a failed download (default: 3)"
    )
    args = parser.parse_args()
    download_rubin_data(args)
