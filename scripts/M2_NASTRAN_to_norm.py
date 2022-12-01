import glob
import os
import pickle

import batoid
from galsim.zernike import Zernike, zernikeBasis
import numpy as np
from scipy.io import loadmat


def main(args):
    indir = os.path.join(
        args.indir,
        "1M2Bending"
    )
    # Load NASTRAN perturbations
    data = loadmat(
        os.path.join(
            indir,
            "0unitLoadCases",
            "T1T2T3.mat"
        )
    )['NXYZap']

    nodeID = data[:, 0]
    x, y, z = data[:, 2:5].T  # Original node positions on M1M3 mirror
    x, y = -y, -x  # x/y flipped and negated

    # NASTRAN perturbations for each mode
    fea_modes = np.array([
        -data[:, 6::3],  # x/y flipped and negated
        -data[:, 5::3],
        -data[:, 7::3]  # z negated
    ]).T
    fea_modes *= 0.0254  # inches -> meters
    x *= 0.0254
    y *= 0.0254
    nmode, nnode, _ = fea_modes.shape

    # Load telescope so we can project perturbations onto surface normal
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    normal_vectors = np.empty((nnode, 3))
    normal_vectors = telescope['M2'].surface.normal(x, y)
    # Do the projection
    normal_modes = np.einsum("abc,bc->ab", fea_modes, normal_vectors)

    # Optionally subtract PTT modes
    if args.M2ptt > 0:
        zbasis = zernikeBasis(
            args.M2ptt, x, y, R_inner=0.9, R_outer=1.71
        )
        for imode in range(nmode):
            coefs, *_ = np.linalg.lstsq(zbasis.T, normal_modes[imode], rcond=None)
            normal_modes[imode] -= Zernike(
                coefs[:4], R_inner=0.9, R_outer=1.71
            )(x, y)

    # Load balanced unit load cases and form pseudo-inverse
    files = sorted(
        glob.glob(
            os.path.join(
                indir,
                "0unitLoadCases",
                "node*.csv"
            )
        )
    )
    C = np.empty((nmode, nmode))
    for i in range(nmode):
        d = np.genfromtxt(files[i], delimiter=',')
        C[:, i] = d[:, 1]
    s = np.linalg.svd(C)[1]
    Cinv = np.linalg.pinv(C, rcond=s[0]*1e-6)

    # Solve for FEA modes
    u, s, vh = np.linalg.svd(Cinv.T@normal_modes, full_matrices=False)

    # Normalize to 1um surface normal deviation
    Udn3norm = np.empty((nnode, nmode))
    Vdn3norm = np.empty((nmode, nmode))
    for imode in range(nmode):
        factor = 1e-6/np.std(vh[imode])
        Udn3norm[:, imode] = vh[imode] * factor
        Vdn3norm[:, imode] = u[:, imode] * factor / s[imode]
    # Note: there are arbitrary minus signs here we don't track.
    with open(args.output, 'wb') as f:
        pickle.dump((x, y, Udn3norm, Vdn3norm), f)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        default="/Users/josh/src/ZEMAX_FEMAP/",
        help=
            "Location of ZEMAX_FEMAP directory.  "
            "Default: /Users/josh/src/ZEMAX_FEMAP/"
    )
    parser.add_argument(
        "--M2ptt",
        type=int,
        default=6,
        help=
            "degree of fit when computing M2 PTT components.  "
            "0 means no PTT subtraction.  "
            "Default: 6"
    )
    parser.add_argument(
        "output",
        default="M2_norm.pkl",
        help=
            "output file name.  "
            "Default: M2_norm.pkl",
        nargs='?'
    )
    args = parser.parse_args()
    main(args)
