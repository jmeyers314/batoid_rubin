import pickle

import batoid


def main(args):
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    with open(args.input, 'rb') as f:
        x, y, Udn3norm, Vdn3norm, coef = pickle.load(f)
    Udn3norm /= telescope['M2'].surface.normal(x, y)[:, 2][:, None]
    with open(args.output, 'wb') as f:
        pickle.dump((x, y, Udn3norm, Vdn3norm), f)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M2_norm.pkl",
    )
    parser.add_argument(
        "output",
        default="M2_sag.pkl",
        help="output file name", nargs='?'
    )
    args = parser.parse_args()
    main(args)
