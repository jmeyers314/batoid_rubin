# Script to create the bend_legacy directory
# for use with batoid_rubin:LSSTBuilder.

import os
from pathlib import Path

import astropy.io.fits as fits
import galsim
import numpy as np
from textwrap import dedent
from scipy.io import loadmat
from astropy.table import Table


def convert_zernikes(bendpath, fn_template, amplitude, outfile):
    # Hard-coding the shape from phosim
    arr = np.empty((20, 29))
    for i in range(20):
        zkfile = bendpath/fn_template.format(i+1)
        zk = np.genfromtxt(zkfile)/amplitude
        # 1-indexed => 0-indexed
        zk = np.concatenate([np.array([0]), zk])
        # Convert to batoid coordinates: x, y = -x, y
        for j in range(1, 29):
            n, m = galsim.zernike.noll_to_zern(j)
            if (n+(m>=0)) % 2 == 0:
                zk[j] *= -1
        # And z = -z, mm to m
        arr[i] = -zk*1e-3
    fits.writeto(
        outfile,
        arr,
        overwrite=True
    )


def convert_grid(bendpath, fn_template, amplitude, outgrid, outcoords):
    out = np.empty((4, 20, 204, 204))
    for i in range(20):
        gridfile = bendpath/fn_template.format(i+1)
        with open(gridfile, 'r') as ff:
            # Read the header
            nx, ny, dx, dy = np.fromstring(ff.readline(), sep=' ')
            nx = int(nx)
            ny = int(ny)
            x = np.arange(nx)*dx*1e-3
            x -= np.mean(x)
            y = np.arange(ny)*dy*1e-3
            y -= np.mean(y)
            # Read rest
            f, dfdx, dfdy, d2fdxdy = np.genfromtxt(ff, unpack=True)
        f.shape = (nx, ny)
        dfdx.shape = (nx, ny)
        dfdy.shape = (nx, ny)
        d2fdxdy.shape = (nx, ny)
        # Zemax convention is 0,0 in first row, not last.  So flipud.
        # Also, since since x, y, z <-> -x, y, -z in Zemax <-> phosim,
        # Need to fliplr and add some minus signs.  Also changing units to m.
        f = -np.fliplr(np.flipud(f.reshape(nx, ny)))*1e-3
        dfdx = np.fliplr(np.flipud(dfdx.reshape(nx, ny)))
        dfdy = -np.fliplr(np.flipud(dfdy.reshape(nx, ny)))
        d2fdxdy = np.fliplr(np.flipud(d2fdxdy.reshape(nx, ny)))*1e3

        out[:, i] = f, dfdx, dfdy, d2fdxdy
    out /= amplitude

    fits.writeto(
        outgrid,
        out,
        overwrite=True
    )
    fits.writeto(
        outcoords,
        np.stack([x, y]),
        overwrite=True
    )


def main(args):
    outpath = Path(args.outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    femappath = Path(args.femap_dir)

    # Load actuator data
    m1m3_actuators = Table(np.genfromtxt(
        femappath / "0M1M3Bending" / "ForceActuatorTable.csv",
        names=True,
        delimiter=',',
        defaultfmt="d,d,f,f,f,f,f,f,f",
        usecols=range(5)
    ))
    m1m3_actuators['Index'] = m1m3_actuators['Index'].astype(int)
    m1m3_actuators['ID'] = m1m3_actuators['ID'].astype(int)
    m1m3_actuators.write(
        outpath/"M1M3_actuator_table.fits.gz",
        overwrite=True
    )

    m2_actuators = Table(np.genfromtxt(
        femappath / "1M2Bending" / "M2actuatorNodesXY_M2_FEA_CS.txt",
        names=['id', 'x', 'y']
    ))
    m2_actuators['id'] = m2_actuators['id'].astype(int)
    # M2 FEA CS -> M2 CS; see https://sitcomtn-003.lsst.io/#m2
    m2_actuators['x'], m2_actuators['y'] = -m2_actuators['y'], -m2_actuators['x']
    m2_actuators['x'] *= 0.0254  # inches -> meters
    m2_actuators['y'] *= 0.0254
    m2_actuators['Index'] = np.arange(len(m2_actuators)).astype(int)
    m2_actuators['ID'] = m2_actuators['id']
    m2_actuators['X_Position'] = m2_actuators['x']
    m2_actuators['Y_Position'] = m2_actuators['y']
    del m2_actuators['id'], m2_actuators['x'], m2_actuators['y']
    m2_actuators.write(
        outpath/"M2_actuator_table.fits.gz",
        overwrite=True
    )

    m1m3_mat = loadmat(
        femappath / "0M1M3Bending" / "2bendingModes" / "m1m3_Urt3norm.mat"
    )
    m1m3_forces = m1m3_mat['Vrt3norm'].T
    m1m3_forces[19] = m1m3_forces[26]
    m1m3_forces = m1m3_forces[:20]

    fits.writeto(
        outpath/"M1M3_bend_forces.fits.gz",
        m1m3_forces,
        overwrite=True
    )

    m2_mat = loadmat(
        femappath / "1M2Bending" / "2bendingModes" / "m2_Urt3norm.mat"
    )
    m2_forces = m2_mat['Vrt3norm']
    m2_forces[[17,18,19]] = m2_forces[[25,26,27]]
    m2_forces = m2_forces[:20]

    fits.writeto(
        outpath/"M2_bend_forces.fits.gz",
        m2_forces,
        overwrite=True
    )

    # First construct the controlling yaml file.
    config = dedent("""
        M1:
          zk:
            file: M1M3_bend_zk.fits.gz
            R_outer: 4.18
          grid:
            file: M1_bend_grid.fits.gz
            coords: M1_bend_coords.fits.gz
        M2:
          zk:
            file: M2_bend_zk.fits.gz
            R_outer: 1.71
          grid:
            file: M2_bend_grid.fits.gz
            coords: M2_bend_coords.fits.gz
        M3:
          zk:
            file: M1M3_bend_zk.fits.gz    # Same as M1!
            R_outer: 4.18
          grid:
            file: M3_bend_grid.fits.gz
            coords: M3_bend_coords.fits.gz
        use_m1m3_modes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        use_m2_modes:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        """
    )
    with open(os.path.join(args.outdir, "bend.yaml"), "w") as f:
        f.write(config[1:])    # Note: Get rid of leading empty line with [:1]

    bendpath = femappath/"2senM"/"0bendingModeGridFiles"

    convert_zernikes(
        bendpath,
        "M13_b{}_0.50_gridz.txt",
        0.5,
        outpath/"M1M3_bend_zk.fits.gz"
    )

    convert_zernikes(
        bendpath,
        "M2_b{}_0.25_gridz.txt",
        0.25,
        outpath/"M2_bend_zk.fits.gz"
    )

    convert_grid(
        bendpath,
        "M1_b{}_0.50_grid.DAT",
        0.5,
        outpath/"M1_bend_grid.fits.gz",
        outpath/"M1_bend_coords.fits.gz"
    )

    convert_grid(
        bendpath,
        "M2_b{}_0.25_grid.DAT",
        0.25,
        outpath/"M2_bend_grid.fits.gz",
        outpath/"M2_bend_coords.fits.gz"
    )

    convert_grid(
        bendpath,
        "M3_b{}_0.50_grid.DAT",
        0.5,
        outpath/"M3_bend_grid.fits.gz",
        outpath/"M3_bend_coords.fits.gz"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--femap_dir",
        help="Path to the ZEMAX_FEMAP directory",
        type=str,
        default="/Users/josh/src/ZEMAX_FEMAP/"
    )
    parser.add_argument(
        "--outdir",
        help="Path to the output directory",
        type=str,
        default="bend_legacy"
    )
    args = parser.parse_args()
    main(args)
