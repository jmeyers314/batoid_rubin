import os
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import asdf


def main(args):
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Read in asdf, copy to dict
    with asdf.open(args.input) as af:
        M1 = dict(grid=dict(), zk=dict())
        M3 = dict(grid=dict(), zk=dict())
        for k1, k2 in (
            ('grid', 'x'),
            ('grid', 'z'),
            ('grid', 'dzdx'),
            ('grid', 'dzdy'),
            ('grid', 'd2zdxy'),
            ('zk', 'coefs')
        ):
            M1[k1][k2] = np.array(af['M1'][k1][k2])
            M3[k1][k2] = np.array(af['M3'][k1][k2])

    for mirror in (M1, M3):
        for k1, k2 in (
            ('grid', 'z'),
            ('grid', 'dzdx'),
            ('grid', 'dzdy'),
            ('grid', 'd2zdxy'),
            ('zk', 'coefs')
        ):
            if tstart:=eval(args.swap):
                tend = np.roll(tstart, len(tstart)//2)
                arr = mirror[k1][k2]
                arr[tstart] = arr[tend]
            # Truncate to nkeep
            mirror[k1][k2] = mirror[k1][k2][:args.nkeep]

    for mirror, name in [(M1, 'M1'), (M3, 'M3')]:
        fits.writeto(
            os.path.join(
                args.outdir,
                f"{name}_bend_coords.fits.gz"
            ),
            np.stack([mirror['grid']['x'], mirror['grid']['x']]),
            overwrite=True
        )
        fits.writeto(
            os.path.join(
                args.outdir,
                f"{name}_bend_grid.fits.gz"
            ),
            np.stack([
                mirror['grid']['z'],
                mirror['grid']['dzdx'],
                mirror['grid']['dzdy'],
                mirror['grid']['d2zdxy']
            ]),
            overwrite=True
        )
        fits.writeto(
            os.path.join(
                args.outdir,
                f"{name}_bend_zk.fits.gz"
            ),
            mirror['zk']['coefs'],
            overwrite=True
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M1M3_decomposition.asdf",
        help=
            "Input Zernike+grid decomposition asdf.  "
            "Default: M1M3_decomposition.asdf"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="batoid_bend/",
    )
    parser.add_argument(
        "--swap",
        type=str,
        # default="[19,26]",
        default="[]",
        help=
            "Swap first half of given modes with second half.  "
            "Default: []"
            # "Default: [19,26]"
    )
    parser.add_argument(
        "--nkeep",
        type=int,
        default=20,
        help="Number of modes to keep.  Default: 20"
    )
    args = parser.parse_args()
    main(args)
