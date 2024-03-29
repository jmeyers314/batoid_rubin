from pathlib import Path

import batoid
from galsim.zernike import Zernike, zernikeBasis
import numpy as np
from scipy.io import loadmat
from astropy.table import Table
import asdf
from tqdm import tqdm


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

    tree = dict()
    tree['actuators'] = actuators
    nactuators = len(actuators)

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

    fea_nodes = Table()
    fea_nodes['nodeID'] = nodeID
    fea_nodes['X_Position'] = x
    fea_nodes['Y_Position'] = y
    fea_nodes['Z_Position'] = z
    tree['fea_nodes'] = fea_nodes
    tree['unit_load'] = dict()
    tree['unit_load']['displacements'] = fea_modes

    # Load unit load forces
    C = np.full((nmode, nactuators), np.nan)
    for actuator in actuators:
        file = indir/"0unitLoadCases"/f"node500{actuator['ID']}.csv"
        C[actuator['Index']] = np.genfromtxt(file, delimiter=',')[:, 1]

    # Check that C is balanced in each row
    np.testing.assert_allclose(np.sum(C, axis=1), 0.0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(C@actuators['X_Position'], 0.0, atol=1e-3, rtol=0)
    np.testing.assert_allclose(C@actuators['Y_Position'], 0.0, atol=1e-3, rtol=0)

    # Compute pseudo-inverse for later
    s = np.linalg.svd(C)[1]
    Cinv = np.linalg.pinv(C, rcond=s[0]*1e-6)

    # Add to tree
    tree['unit_load']['forces'] = C

    # Load telescope so we can project perturbations onto surface normal
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    normal_vectors = np.empty((nnode, 3))
    normal_vectors[w1] = telescope['M1'].surface.normal(x[w1], y[w1])
    normal_vectors[w3] = telescope['M3'].surface.normal(x[w3], y[w3])

    tree['M1M3_normal_vectors'] = normal_vectors

    # Do the projection
    normal_modes = np.einsum("abc,bc->ab", fea_modes, normal_vectors)
    tree['unit_load']['normal_displacement'] = normal_modes

    ptt_modes = np.array(normal_modes)

    # Optionally subtract PTT modes
    if args.M1ptt > 0:
        zbasis1 = zernikeBasis(
            args.M1ptt, x[w1], y[w1], R_inner=M1_inner, R_outer=M1_outer
        )
        for imode in range(nmode):
            coefs, *_ = np.linalg.lstsq(zbasis1.T, ptt_modes[imode, w1], rcond=s[0]*1e-6)
            m1_correction = Zernike(
                coefs[:4], R_inner=M1_inner, R_outer=M1_outer
            )(x, y)
            ptt_modes[imode] -= m1_correction
    if args.M3ptt > 0:
        zbasis3 = zernikeBasis(
            args.M3ptt, x[w3], y[w3], R_inner=M3_inner, R_outer=M3_outer
        )
        for imode in range(nmode):
            coefs, *_ = np.linalg.lstsq(zbasis3.T, ptt_modes[imode, w3], rcond=None)
            m3_correction = Zernike(
                coefs[:4], R_inner=M3_inner, R_outer=M3_outer
            )(x[w3], y[w3])
            ptt_modes[imode, w3] -= m3_correction

    sag_modes = np.array(ptt_modes)
    sag_modes[:, w1] /= telescope['M1'].surface.normal(x[w1], y[w1])[:, 2]
    sag_modes[:, w3] /= telescope['M3'].surface.normal(x[w3], y[w3])[:, 2]

    tree['unit_load']['ptt_normal_displacement'] = ptt_modes

    mat = loadmat(indir / "2bendingModes" / "m1m3_Urt3norm.mat")
    target_forces =  mat['Vrt3norm'].T

    # Normalize to 1um RMS surface deviation.
    normal_modes_1um = np.empty((nmode, nnode))
    force_modes_1um = np.empty((nmode, nactuators))
    coef_1um = np.empty((nmode, nmode))  # coefs of unit load force or ptt deviation vectors

    if args.validate:
        iterator = range(nmode)
    else:
        iterator = tqdm(range(nmode))
    for imode in iterator:
        weights, *_ = np.linalg.lstsq(C.T, target_forces[imode], rcond=None)
        np.testing.assert_allclose(weights@C, target_forces[imode], atol=1e-6, rtol=0)
        norm_mode = weights@ptt_modes
        if args.validate:
            sag_mode = weights@sag_modes
            print(
                f"{imode:3d}"
                f" {np.std(sag_mode):10.2e}"
                f" {np.ptp(sag_mode):10.2e}  {np.ptp(sag_mode - mat['Urt3norm'][:, imode]):10.2e}"
                f" {np.ptp(sag_mode)/np.ptp(sag_mode - mat['Urt3norm'][:, imode]):10.2f}"
                f" {np.ptp(sag_mode)/np.quantile(np.abs(sag_mode - mat['Urt3norm'][:, imode]), 0.9):10.2f}"
            )
        factor = 1e-6/np.std(norm_mode)
        normal_modes_1um[imode] = factor*norm_mode
        force_modes_1um[imode] = factor*weights@C
        coef_1um[imode] = factor*weights

    influence = Cinv@ptt_modes
    tree['influence_normal_displacement'] = influence

    tree['bend_1um'] = dict()
    tree['bend_1um']['normal'] = normal_modes_1um
    tree['bend_1um']['force'] = force_modes_1um
    tree['bend_1um']['coef'] = coef_1um

    # Convert displacement along normal to along sag
    sag_modes_1um = np.array(normal_modes_1um)
    sag_modes_1um[:, w1] /= telescope['M1'].surface.normal(x[w1], y[w1])[:, 2]
    sag_modes_1um[:, w3] /= telescope['M3'].surface.normal(x[w3], y[w3])[:, 2]

    tree['bend_1um']['sag'] = sag_modes_1um
    tree['M1_nodes'] = w1
    tree['M3_nodes'] = w3
    tree['args'] = vars(args)

    # Note: there are arbitrary minus signs here we don't track.
    # We might also have some mode ordering swaps when switching PTT off and on.
    asdf.AsdfFile(tree).write_to(args.output)


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
        default="M1M3_NASTRAN.asdf",
        help=
            "output file name.  "
            "Default: M1M3_NASTRAN.asdf",
        nargs='?'
    )
    parser.add_argument(
        "--validate",
        action='store_true',
    )
    args = parser.parse_args()
    main(args)
