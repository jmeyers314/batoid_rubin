from pathlib import Path
from subprocess import run


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
    cmd = f"zenodo_get {DOI}"
    if not args.outdir:
        outdir = Path(__file__).parent / args.dataset
    else:
        outdir = Path(args.outdir)
    cmd += f" -o {outdir}"
    run(cmd, shell=True)


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
