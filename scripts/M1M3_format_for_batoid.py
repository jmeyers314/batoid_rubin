import os
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import pickle

def main(args):
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    with open(args.input, 'rb') as f:
        (M1zk, M3zk, M1M3zk,
         M1_x_grid, M1_z_grid, M1_dzdx_grid, M1_dzdy_grid, M1_d2zdxy_grid,
         M3_x_grid, M3_z_grid, M3_dzdx_grid, M3_dzdy_grid, M3_d2zdxy_grid
        ) = pickle.load(f)

    fits.writeto(
        os.path.join(
            args.outdir,
            "M1_bend_coords.fits"
        ),
        np.stack([M1_x_grid, M1_x_grid]),
        overwrite=True
    )
    fits.writeto(
        os.path.join(
            args.outdir,
            "M3_bend_coords.fits"
        ),
        np.stack([M3_x_grid, M3_x_grid]),
        overwrite=True
    )
    fits.writeto(
        os.path.join(
            args.outdir,
            "M1_bend_grid.fits"
        ),
        np.stack([M1_z_grid, M1_dzdx_grid, M1_dzdy_grid, M1_d2zdxy_grid]),
        overwrite=True
    )
    fits.writeto(
        os.path.join(
            args.outdir,
            "M3_bend_grid.fits"
        ),
        np.stack([M3_z_grid, M3_dzdx_grid, M3_dzdy_grid, M3_d2zdxy_grid]),
        overwrite=True
    )
    if np.all(M1zk):
        fits.writeto(
            os.path.join(
                args.outdir,
                "M13_bend_zk.fits"
            ),
            M1M3zk,
            overwrite=True
        )
    else:
        fits.writeto(
            os.path.join(
                args.outdir,
                "M1_bend_zk.fits"
            ),
            M1zk,
            overwrite=True
        )
        fits.writeto(
            os.path.join(
                args.outdir,
                "M3_bend_zk.fits"
            ),
            M3zk,
            overwrite=True
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M1M3_decomposition.pkl",
        help=
            "Input Zernike+grid decomposition pkl.  "
            "Default: M1M3_decomposition.pkl"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="batoid_bend/",
    )
    args = parser.parse_args()
    main(args)
