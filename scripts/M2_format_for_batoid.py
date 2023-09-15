import os
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import asdf


def main(args):
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

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
        os.path.join(
            args.outdir,
            f"M2_bend_coords.fits.gz"
        ),
        np.stack([M2['grid']['x'], M2['grid']['x']]),
        overwrite=True
    )
    fits.writeto(
        os.path.join(
            args.outdir,
            f"M2_bend_grid.fits.gz"
        ),
        np.stack([
            M2['grid']['z'],
            M2['grid']['dzdx'],
            M2['grid']['dzdy'],
            M2['grid']['d2zdxy']
        ]),
        overwrite=True
    )
    fits.writeto(
        os.path.join(
            args.outdir,
            f"M2_bend_zk.fits.gz"
        ),
        M2['zk']['coefs'],
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
    args = parser.parse_args()
    main(args)
