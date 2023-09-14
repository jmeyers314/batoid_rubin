import pickle

import batoid


def main(args):
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    with open(args.input, 'rb') as f:
        x, y, w1, w3, Udn3norm, Vdn3norm, coef = pickle.load(f)
    Udn3norm[w1] /= telescope['M1'].surface.normal(x[w1], y[w1])[:, 2][:, None]
    Udn3norm[w3] /= telescope['M3'].surface.normal(x[w3], y[w3])[:, 2][:, None]
    with open(args.output, 'wb') as f:
        pickle.dump((x, y, w1, w3, Udn3norm, Vdn3norm), f)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M1M3_norm.pkl",
    )
    parser.add_argument(
        "output",
        default="M1M3_sag.pkl",
        help="output file name", nargs='?'
    )
    args = parser.parse_args()
    main(args)
