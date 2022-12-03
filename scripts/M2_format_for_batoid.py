import os
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import pickle

def main(args):
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    with open(args.input, 'rb') as f:
        (M2zk,
         M2_x_grid, M2_z_grid, M2_dzdx_grid, M2_dzdy_grid, M2_d2zdxy_grid,
        ) = pickle.load(f)

    if tstart:=eval(args.swap):
        # swap first half for second half
        tend = np.roll(tstart, len(tstart)//2)
        for arr in (
            M2zk,
            M2_z_grid, M2_dzdx_grid, M2_dzdy_grid, M2_d2zdxy_grid
        ):
            arr[tstart] = arr[tend]

    M2zk = M2zk[:args.nkeep]

    M2_z_grid = M2_z_grid[:args.nkeep]
    M2_dzdx_grid = M2_dzdx_grid[:args.nkeep]
    M2_dzdy_grid = M2_dzdy_grid[:args.nkeep]
    M2_d2zdxy_grid = M2_d2zdxy_grid[:args.nkeep]

    fits.writeto(
        os.path.join(
            args.outdir,
            "M2_bend_coords.fits.gz"
        ),
        np.stack([M2_x_grid, M2_x_grid]),
        overwrite=True
    )
    fits.writeto(
        os.path.join(
            args.outdir,
            "M2_bend_grid.fits.gz"
        ),
        np.stack([M2_z_grid, M2_dzdx_grid, M2_dzdy_grid, M2_d2zdxy_grid]),
        overwrite=True
    )
    fits.writeto(
        os.path.join(
            args.outdir,
            "M2_bend_zk.fits.gz"
        ),
        M2zk,
        overwrite=True
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M2_decomposition.pkl",
        help=
            "Input Zernike+grid decomposition pkl.  "
            "Default: M2_decomposition.pkl"
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
