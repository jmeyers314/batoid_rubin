from pathlib import Path
import pickle

import batoid
from galsim.zernike import Zernike, zernikeBasis
import numpy as np
from scipy.io import loadmat
from astropy.table import Table


M1_outer = 4.18
M1_inner = 2.558
M2_outer = 1.71
M2_inner = 0.9
M3_outer = 2.508
M3_inner = 0.55


def main(args):
    indir = Path(args.indir) / "0M1M3Bending"
    # Load actuator data
    actuators = Table(np.genfromtxt(
        indir / "ForceActuatorTable.csv",
        names=True,
        delimiter=',',
        defaultfmt="d,d,f,f,f,f,f,f,f",
        usecols=range(5)
    ))
    actuators['Index'] = actuators['Index'].astype(int)
    actuators['ID'] = actuators['ID'].astype(int)

    # Load NASTRAN perturbations
    t1t2t3 = loadmat(indir / "0unitLoadCases" / "T1T2T3.mat")['NXYZap']
    nodeID = t1t2t3[:, 0]
    x, y, z = t1t2t3[:, 2:5].T  # Original node positions on M1M3 mirror
    dx = t1t2t3[:, 5::3]
    dy = t1t2t3[:, 6::3]
    dz = t1t2t3[:, 7::3]
    fea_modes = np.array([dx, dy, dz]).T
    w1 = nodeID < 800000
    w3 = nodeID >= 800000
    nmode, nnode, _ = fea_modes.shape

    # Load unit load forces
    C = np.full((nmode, nmode), np.nan)
    for actuator in actuators:
        file = indir/"0unitLoadCases"/f"node500{actuator['ID']}.csv"
        C[actuator['Index']] = np.genfromtxt(file, delimiter=',')[:, 1]

    # Compute pseudo-inverse for later
    s = np.linalg.svd(C)[1]
    Cinv = np.linalg.pinv(C, rcond=s[0]*1e-6)

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

    # Solve for FEA modes
    u, s, vh = np.linalg.svd(Cinv@normal_modes, full_matrices=False)

    # Normalize to 1um RMS surface deviation.
    Udn3norm = np.empty((nnode, nmode))  # surface deviations
    Vdn3norm = np.empty((nmode, nmode))  # actuator forces
    coef = np.empty((nmode, nmode))  # coefs of unit load force or deviation vectors
    for imode in range(nmode):
        factor = 1e-6/np.std(vh[imode])
        Udn3norm[:, imode] = vh[imode] * factor
        Vdn3norm[:, imode] = u[:, imode] * factor / s[imode]
        coef[:, imode] = Cinv@Vdn3norm[:, imode]

        if imode < 30:  # mostly care about first ~30 modes
            np.testing.assert_allclose(
                coef[:, imode]@normal_modes, Udn3norm[:, imode],
                atol=2e-10, rtol=0.0
            )

            np.testing.assert_allclose(
                coef[:, imode]@C, Vdn3norm[:, imode],
                atol=1e-3, rtol=1e-3
            )

    Udn3sag = np.array(Udn3norm)
    Udn3sag[w1] /= telescope['M1'].surface.normal(x[w1], y[w1])[:, 2][:, None]
    Udn3sag[w3] /= telescope['M3'].surface.normal(x[w3], y[w3])[:, 2][:, None]

    # Note: there are arbitrary minus signs here we don't track.
    # We might also have some mode ordering swaps when switching PTT off and on.
    # These all appear to be in the batoid coordinate system here.
    with open(args.output, 'wb') as f:
        pickle.dump((x, y, w1, w3, Udn3norm, Vdn3norm, Udn3sag, coef), f)


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
