import os

import astropy.io.fits as fits
import numpy as np
from scipy.interpolate import Rbf


def main(args):
    # M1M3_1um_156_grid, force
    # Txt files are more recent, so using those
    # arr1 = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_1um_156_grid.npy"))  # Is not PTT-subtracted!
    arr = np.loadtxt(os.path.join(args.indir, "data", "M1M3", "M1M3_1um_156_grid.txt"))  # Is PTT-subtracted!
    m13x = arr[:, 1]
    m13y = arr[:, 2]
    fits.writeto(os.path.join(args.outdir, "M1M3_1um_156_grid.fits.gz"), arr, overwrite=True)
    arr = np.loadtxt(os.path.join(args.indir, "data", "M1M3", "M1M3_1um_156_force.txt"))
    fits.writeto(os.path.join(args.outdir, "M1M3_1um_156_force.fits.gz"), arr, overwrite=True)

    # M1M3_dxdydz_zenith, horizon
    arr = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_dxdydz_zenith.npy"))
    fits.writeto(os.path.join(args.outdir, "M1M3_dxdydz_zenith.fits.gz"), arr, overwrite=True)
    arr = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_dxdydz_horizon.npy"))
    fits.writeto(os.path.join(args.outdir, "M1M3_dxdydz_horizon.fits.gz"), arr, overwrite=True)

    # M1M3_thermal_FEA
    arr = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_thermal_FEA.npy"))
    # One time interpolation here
    tx = arr[:, 0]  # radius normalized in M1M3 CS
    ty = arr[:, 1]
    tbdz = Rbf(tx, ty, arr[:, 2])(m13x/4.18, m13y/4.18)
    txdz = Rbf(tx, ty, arr[:, 3])(m13x/4.18, m13y/4.18)
    tydz = Rbf(tx, ty, arr[:, 4])(m13x/4.18, m13y/4.18)
    tzdz = Rbf(tx, ty, arr[:, 5])(m13x/4.18, m13y/4.18)
    trdz = Rbf(tx, ty, arr[:, 6])(m13x/4.18, m13y/4.18)
    fits.writeto(
        os.path.join(args.outdir, "M1M3_thermal_FEA.fits.gz"),
        np.stack([tbdz, txdz, tydz, tzdz, trdz]),
        overwrite=True
    )

    # M1M3_force_zenith, horizon
    arr = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_force_zenith.npy"))
    fits.writeto(os.path.join(args.outdir, "M1M3_force_zenith.fits.gz"), arr, overwrite=True)
    arr = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_force_horizon.npy"))
    fits.writeto(os.path.join(args.outdir, "M1M3_force_horizon.fits.gz"), arr, overwrite=True)

    # M1M3_influence_256
    arr = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_influence_256.npy"))
    fits.writeto(os.path.join(args.outdir, "M1M3_influence_256.fits.gz"), arr, overwrite=True)

    # M1M3_LUT
    arr = np.loadtxt(os.path.join(args.indir, "data", "M1M3", "M1M3_LUT.txt"))
    fits.writeto(os.path.join(args.outdir, "M1M3_LUT.fits.gz"), arr, overwrite=True)

    # M1M3_1000N_UL_shape_156
    arr = np.load(os.path.join(args.indir, "data", "M1M3", "M1M3_1000N_UL_shape_156.npy"))
    fits.writeto(os.path.join(args.outdir, "M1M3_1000N_UL_shape_156.fits.gz"), arr, overwrite=True)

    # M2_1um_grid, force
    arr = np.loadtxt(os.path.join(args.indir, "data", "M2", "M2_1um_grid.DAT"))  # Is PTT-subtracted!
    # Save these for below
    m2x = arr[:, 1]  # meters
    m2y = arr[:, 2]
    fits.writeto(os.path.join(args.outdir, "M2_1um_grid.fits.gz"), arr, overwrite=True)
    arr = np.loadtxt(os.path.join(args.indir, "data", "M2", "M2_1um_force.DAT"))  # Is PTT-subtracted!
    fits.writeto(os.path.join(args.outdir, "M2_1um_force.fits.gz"), arr, overwrite=True)

    # One time interpolation here
    arr = np.loadtxt(os.path.join(args.indir, "data", "M2", "M2_GT_FEA.txt"), skiprows=1)  # Is PTT-subtracted!
    tx = arr[:, 0]  # Radius-normalized coordinates in M2 CS
    ty = arr[:, 1]
    zdz = Rbf(tx, ty, arr[:, 2])(m2x/1.71, m2y/1.71)
    hdz = Rbf(tx, ty, arr[:, 3])(m2x/1.71, m2y/1.71)
    tzdz = Rbf(tx, ty, arr[:, 4])(m2x/1.71, m2y/1.71)
    trdz = Rbf(tx, ty, arr[:, 5])(m2x/1.71, m2y/1.71)
    fits.writeto(
        os.path.join(args.outdir, "M2_GT_FEA.fits.gz"),
        np.stack([zdz, hdz, tzdz, trdz]),
        overwrite=True
    )

    # Worry about camera later...


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "indir",
        type=str,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="fea",
    )
    args = parser.parse_args()
    main(args)