from pathlib import Path

import batoid
from galsim.zernike import Zernike, zernikeBasis
import numpy as np
from scipy.io import loadmat
from astropy.table import Table
import asdf


M1_outer = 4.18
M1_inner = 2.558
M2_outer = 1.71
M2_inner = 0.9
M3_outer = 2.508
M3_inner = 0.55


def main(args):
    indir = Path(args.indir) / "1M2Bending"

    actuators = Table(np.genfromtxt(
        indir / "M2actuatorNodesXY_M2_FEA_CS.txt",
        names=['id', 'x', 'y']
    ))
    actuators['id'] = actuators['id'].astype(int)
    # M2 FEA CS -> M2 CS; see https://sitcomtn-003.lsst.io/#m2
    actuators['x'], actuators['y'] = -actuators['y'], -actuators['x']
    actuators['x'] *= 0.0254  # inches -> meters
    actuators['y'] *= 0.0254

    tree = dict()
    tree['actuators'] = actuators
    nactuators = len(actuators)

    # Load NASTRAN perturbations
    t1t2t3 = loadmat(indir / "0unitLoadCases" / "T1T2T3.mat")['NXYZap']
    nodeID = t1t2t3[:, 0]
    t1t2t3[:, 2:] *= 0.0254 # inches -> meters
    # M2 FEA CS -> M2 CS here too; x, y, z => -y, -x, -z
    x = -t1t2t3[:, 3]
    y = -t1t2t3[:, 2]
    z = -t1t2t3[:, 4]
    dx = -t1t2t3[:, 6::3]
    dy = -t1t2t3[:, 5::3]
    dz = -t1t2t3[:, 7::3]
    fea_modes = np.array([dx, dy, dz]).T
    nmode, nnode, _ = fea_modes.shape

    fea_nodes = Table()
    fea_nodes['nodeID'] = nodeID
    fea_nodes['X_Position'] = x
    fea_nodes['Y_Position'] = y
    fea_nodes['Z_Position'] = z
    tree['fea_nodes'] = fea_nodes
    tree['unit_load'] = dict()
    tree['unit_load']['displacements'] = fea_modes

    # Load telescope so we can project perturbations onto surface normal
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    normal_vectors = np.empty((nnode, 3))
    normal_vectors = telescope['M2'].surface.normal(x, y)
    # Do the projection
    normal_modes = np.einsum("abc,bc->ab", fea_modes, normal_vectors)

    ptt_modes = np.array(normal_modes)

    tree['M2_normal_vectors'] = normal_vectors
    tree['unit_load']['normal_displacement'] = normal_modes

    # Optionally subtract PTT modes
    if args.M2ptt > 0:
        zbasis = zernikeBasis(
            args.M2ptt, x, y, R_inner=M2_inner, R_outer=M2_outer
        )
        for imode in range(nmode):
            coefs, *_ = np.linalg.lstsq(zbasis.T, ptt_modes[imode], rcond=None)
            ptt_modes[imode] -= Zernike(
                coefs[:4], R_inner=M2_inner, R_outer=M2_outer
            )(x, y)

    tree['unit_load']['ptt_normal_displacement'] = ptt_modes

    # Load balanced unit load cases and form pseudo-inverse
    C = np.full((nmode, nmode), np.nan)
    for i, actuator in enumerate(actuators):
        file = indir/"0unitLoadCases"/f"node{actuator['id']}.csv"
        C[i] = np.genfromtxt(file, delimiter=',')[:, 1]

    tree['unit_load']['forces'] = C

    s = np.linalg.svd(C)[1]
    Cinv = np.linalg.pinv(C, rcond=s[0]*1e-6)

    influence = Cinv@ptt_modes
    tree['influence_normal_displacement'] = influence

    # Solve for FEA modes
    u, s, vh = np.linalg.svd(influence, full_matrices=False)

    # Normalize to 1um surface normal deviation
    normal_modes_1um = np.empty((nmode, nnode))
    force_modes_1um = np.empty((nmode, nactuators))
    coef_1um = np.empty((nmode, nmode))  # coefs of unit load force or deviation vectors


    Udn3norm = np.empty((nnode, nmode))  # surface deviations
    Vdn3norm = np.empty((nmode, nmode))  # actuator forces
    coef = np.empty((nmode, nmode))  # coefs of unit load force or deviation vectors
    for imode in range(nmode):
        factor = 1e-6/np.std(vh[imode])
        normal_modes_1um[imode] = vh[imode] * factor
        force_modes_1um[imode] = u.T[imode] * factor / s[imode]
        # coef_1um[imode], *_ = np.linalg.lstsq(ptt_modes.T, normal_modes_1um[imode], rcond=None)
        coef_1um[imode] = Cinv@force_modes_1um[imode]

        if imode < 30:  # mostly care about first ~30 modes
            np.testing.assert_allclose(
                coef_1um[imode]@ptt_modes, normal_modes_1um[imode],
                atol=1e-9, rtol=0.0
            )

            np.testing.assert_allclose(
                coef_1um[imode]@C, force_modes_1um[imode],
                atol=1e-3, rtol=1e-3
            )

    tree['bend_1um'] = dict()
    tree['bend_1um']['normal'] = normal_modes_1um
    tree['bend_1um']['force'] = force_modes_1um
    tree['bend_1um']['coef'] = coef_1um

    # Convert displacement along normal to along sag
    sag_modes_1um = np.array(normal_modes_1um)
    sag_modes_1um /= telescope['M2'].surface.normal(x, y)[:, 2]

    tree['bend_1um']['sag'] = sag_modes_1um
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
        default="M2_norm.asdf",
        help=
            "output file name.  "
            "Default: M2_norm.asdf",
        nargs='?'
    )
    args = parser.parse_args()
    main(args)
