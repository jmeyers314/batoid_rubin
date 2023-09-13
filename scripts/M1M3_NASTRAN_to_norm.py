import glob
import os
import pickle

import batoid
from galsim.zernike import Zernike, zernikeBasis
import numpy as np
from scipy.io import loadmat


M1_outer = 4.18
M1_inner = 2.558
M2_outer = 1.71
M2_inner = 0.9
M3_outer = 2.508
M3_inner = 0.55


def main(args):
    indir = os.path.join(
        args.indir,
        "0M1M3Bending"
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
    # NASTRAN perturbations for each mode
    fea_modes = np.array([
        data[:, 5::3],
        data[:, 6::3],
        data[:, 7::3]
    ]).T
    w1 = nodeID < 800000
    w3 = nodeID >= 800000
    nmode, nnode, _ = fea_modes.shape

    # Load telescope so we can project perturbations onto surface normal
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    normal_vectors = np.empty((nnode, 3))
    normal_vectors[w1] = telescope['M1'].surface.normal(x[w1], y[w1])
    normal_vectors[w3] = telescope['M3'].surface.normal(x[w3], y[w3])
    # Do the projection
    normal_modes = np.einsum("abc,bc->ab", fea_modes, normal_vectors)

    # Optionally subtract PTT modes
    if args.M1ptt > 0:
        zbasis1 = zernikeBasis(
            args.M1ptt, x[w1], y[w1], R_inner=M1_inner, R_outer=M1_outer
        )
        for imode in range(nmode):
            coefs, *_ = np.linalg.lstsq(zbasis1.T, normal_modes[imode, w1], rcond=None)
            m1_correction = Zernike(
                coefs[:4], R_inner=M1_inner, R_outer=M1_outer
            )(x, y)
            normal_modes[imode] -= m1_correction
    if args.M3ptt > 0:
        zbasis3 = zernikeBasis(
            args.M3ptt, x[w3], y[w3], R_inner=M3_inner, R_outer=M3_outer
        )
        for imode in range(nmode):
            coefs, *_ = np.linalg.lstsq(zbasis3.T, normal_modes[imode, w3], rcond=None)
            m3_correction = Zernike(
                coefs[:4], R_inner=M3_inner, R_outer=M3_outer
            )(x[w3], y[w3])
            normal_modes[imode, w3] -= m3_correction

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

    # Normalize to 1um RMS surface deviation.
    Udn3norm = np.empty((nnode, nmode))
    Vdn3norm = np.empty((nmode, nmode))
    for imode in range(nmode):
        factor = 1e-6/np.std(vh[imode])
        Udn3norm[:, imode] = vh[imode] * factor
        Vdn3norm[:, imode] = u[:, imode] * factor / s[imode]

    # Note: there are arbitrary minus signs here we don't track.
    # We might also have some position swaps when switching PTT off and on.
    # These all appear to be in the batoid coordinate system here.
    with open(args.output, 'wb') as f:
        pickle.dump((x, y, w1, w3, Udn3norm, Vdn3norm), f)


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
        "--M1ptt",
        type=int,
        default=6,
        help=
            "degree of fit when computing M1 PTT components.  "
            "0 means no PTT subtraction.  "
            "The fit to M1 PTT will be removed from both M1 and M3.  "
            "Default: 6"
    )
    parser.add_argument(
        "--M3ptt",
        type=int,
        default=0,
        help=
            "degree of fit when computing M3 PTT components.  "
            "0 means no PTT subtraction.  "
            "Only M3 will be affected by this correction.  "
            "Default: 0"
    )
    parser.add_argument(
        "output",
        default="M1M3_norm.pkl",
        help=
            "output file name.  "
            "Default: M1M3_norm.pkl",
        nargs='?'
    )
    args = parser.parse_args()
    main(args)
