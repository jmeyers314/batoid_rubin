import os
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import asdf


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read in asdf, copy to dict
    with asdf.open(args.input) as af:
        M2 = dict(grid=dict(), zk=dict())
        for k1, k2 in (
            ('grid', 'x'),
            ('grid', 'z'),
            ('grid', 'dzdx'),
            ('grid', 'dzdy'),
            ('grid', 'd2zdxy'),
            ('zk', 'coefs')
        ):
            M2[k1][k2] = np.array(af['M2'][k1][k2])
        forces_file = af['args']['input']

    for k1, k2 in (
        ('grid', 'z'),
        ('grid', 'dzdx'),
        ('grid', 'dzdy'),
        ('grid', 'd2zdxy'),
        ('zk', 'coefs')
    ):
        if tstart:=eval(args.swap):
            tend = np.roll(tstart, len(tstart)//2)
            arr = M2[k1][k2]
            arr[tstart] = arr[tend]
        # Truncate to nkeep
        M2[k1][k2] = M2[k1][k2][:args.nkeep]

    fits.writeto(
        outdir / f"M2_bend_coords.fits.gz",
        np.stack([M2['grid']['x'], M2['grid']['x']]),
        overwrite=True
    )
    fits.writeto(
        outdir / f"M2_bend_grid.fits.gz",
        np.stack([
            M2['grid']['z'],
            M2['grid']['dzdx'],
            M2['grid']['dzdy'],
            M2['grid']['d2zdxy']
        ]),
        overwrite=True
    )
    fits.writeto(
        outdir / f"M2_bend_zk.fits.gz",
        M2['zk']['coefs'],
        overwrite=True
    )

    if args.do_forces:
        with asdf.open(forces_file) as af:
            fits.writeto(
                outdir / "M2_bend_forces.fits.gz",
                af['bend_1um']['force'][:args.nkeep],
                overwrite=True
            )

            m2_actuators = af['actuators']
            m2_actuators['Index'] = np.arange(len(m2_actuators)).astype(int)
            m2_actuators['ID'] = m2_actuators['id']
            m2_actuators['X_Position'] = m2_actuators['x']
            m2_actuators['Y_Position'] = m2_actuators['y']
            del m2_actuators['id'], m2_actuators['x'], m2_actuators['y']
            m2_actuators.write(
                outdir/"M2_actuator_table.fits.gz",
                overwrite=True
            )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M2_decomposition.asdf",
        help=
            "Input Zernike+grid decomposition asdf.  "
            "Default: M2_decomposition.asdf"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="batoid_bend/",
    )
    parser.add_argument(
        "--swap",
        type=str,
        # default="[17,18,19,25,26,27]",
        default="[]",
        help=
            "Swap first half of given modes with second half.  "
            "Default: []"
            # "Default: [17,18,19,25,26,27]"
    )
    parser.add_argument(
        "--nkeep",
        type=int,
        default=20,
        help="Number of modes to keep.  Default: 20"
    )
    parser.add_argument(
        "--do_forces",
        action='store_true',
        help="Populate forces from asdf file.  Default: False"
    )
    args = parser.parse_args()
    main(args)
