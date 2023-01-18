import os
from pathlib import Path

import astropy.io.fits as fits
import galsim
import numpy as np
import yaml


def main(args):
    outpath = Path(args.outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    femappath = Path(args.femap_dir)

    config = {
        'M1': {
            'zk': {
                'file': 'M13_bend_zk.fits.gz',
                'R_outer': 4.18
            },
            'grid': {
                'file': 'M1_bend_grid.fits.gz',
                'coords': 'M1_bend_coords.fits.gz'
            }
        },
        'M2': {
            'zk': {
                'file': 'M2_bend_zk.fits.gz',
                'R_outer': 1.71
            },
            'grid': {
                'file': 'M2_bend_grid.fits.gz',
                'coords': 'M2_bend_coords.fits.gz'
            }
        },
        'M3': {
            'zk': {
                'file': 'M13_bend_zk.fits.gz',  # Same as M1!
                'R_outer': 4.18
            },
            'grid': {
                'file': 'M3_bend_grid.fits.gz',
                'coords': 'M3_bend_coords.fits.gz'
            }
        }
    }
    with open(os.path.join(args.outdir, "bend.yaml"), "w") as f:
        yaml.dump(config, f)

    bendpath = femappath/"2senM"/"0bendingModeGridFiles"

    # M1M3 Zernikes
    m1m3_zk = np.empty((20, 29))
    for i in range(20):
        zkfile = bendpath/f"M13_b{i+1}_0.50_gridz.txt"
        zk = np.genfromtxt(zkfile)/0.5  # scale from 0.5 micron perturbation to 1.0 micron
        # Make 0-indexed
        zk = np.concatenate([np.array([0]), zk])
        # Convert to batoid coordinates: x, y = -x, y
        for j in range(1, 29):
            n, m = galsim.zernike.noll_to_zern(j)
            if (n+(m>=0)) % 2 == 0:  # antisymmetric in x
                zk[j] *= -1
        # And z = -z
        m1m3_zk[i] = -zk*1e-3  # mm to m

    fits.writeto(
        outpath/"M13_bend_zk.fits.gz",
        m1m3_zk,
        overwrite=True
    )

    # M2 Zernikes
    m2_zk = np.empty((20, 29))
    for i in range(20):
        zkfile = bendpath/f"M2_b{i+1}_0.25_gridz.txt"
        zk = np.genfromtxt(zkfile)/0.25  # scale from 0.25 micron perturbation to 1.0 micron
        # Make 0-indexed
        zk = np.concatenate([np.array([0]), zk])
        # Convert to batoid coordinates: x, y = -x, y
        for j in range(1, 29):
            n, m = galsim.zernike.noll_to_zern(j)
            if (n+(m>=0)) % 2 == 0:  # antisymmetric in x
                zk[j] *= -1
        # And z = -z
        m2_zk[i] = -zk*1e-3  # mm to m

    fits.writeto(
        outpath/"M2_bend_zk.fits.gz",
        m2_zk,
        overwrite=True
    )

    # M1 grid
    m1_grid = np.empty((4, 20, 204, 204))
    for i in range(20):
        gridfile = bendpath/f"M1_b{i+1}_0.50_grid.DAT"
        with open(gridfile, 'r') as ff:
            # Read the header
            nx, ny, dx, dy = np.fromstring(ff.readline(), sep=' ')
            nx = int(nx)
            ny = int(ny)
            m1x = np.arange(nx)*dx*1e-3 # mm to m
            m1x -= np.mean(m1x)
            m1y = np.arange(ny)*dy*1e-3 # mm to m
            m1y -= np.mean(m1y)
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

        m1_grid[:, i] = f, dfdx, dfdy, d2fdxdy
    m1_grid /= 0.5  # scale from 0.5 micron perturbation to 1.0 micron

    fits.writeto(
        outpath/"M1_bend_grid.fits.gz",
        m1_grid,
        overwrite=True
    )
    fits.writeto(
        outpath/"M1_bend_coords.fits.gz",
        np.stack([m1x, m1y]),
        overwrite=True
    )

    # M2 grid
    m2_grid = np.empty((4, 20, 204, 204))
    for i in range(20):
        gridfile = bendpath/f"M2_b{i+1}_0.25_grid.DAT"
        with open(gridfile, 'r') as ff:
            # Read the header
            nx, ny, dx, dy = np.fromstring(ff.readline(), sep=' ')
            nx = int(nx)
            ny = int(ny)
            m2x = np.arange(nx)*dx*1e-3 # mm to m
            m2x -= np.mean(m2x)
            m2y = np.arange(ny)*dy*1e-3 # mm to m
            m2y -= np.mean(m2y)
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

        m2_grid[:, i] = f, dfdx, dfdy, d2fdxdy
    m2_grid /= 0.25  # scale from 0.25 micron perturbation to 1.0 micron

    fits.writeto(
        outpath/"M2_bend_grid.fits.gz",
        m2_grid,
        overwrite=True
    )
    fits.writeto(
        outpath/"M2_bend_coords.fits.gz",
        np.stack([m2x, m2y]),
        overwrite=True
    )

    # M3 grid
    m3_grid = np.empty((4, 20, 204, 204))
    for i in range(20):
        gridfile = bendpath/f"M3_b{i+1}_0.50_grid.DAT"
        with open(gridfile, 'r') as f:
            # Read the header
            nx, ny, dx, dy = np.fromstring(f.readline(), sep=' ')
            nx = int(nx)
            ny = int(ny)
            m3x = np.arange(nx)*dx*1e-3 # mm to m
            m3x -= np.mean(m3x)
            m3y = np.arange(ny)*dy*1e-3 # mm to m
            m3y -= np.mean(m3y)
            # Read rest
            f, dfdx, dfdy, d2fdxdy = np.genfromtxt(f, unpack=True)
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

        m3_grid[:, i] = f, dfdx, dfdy, d2fdxdy
    m3_grid /= 0.5  # scale from 0.5 micron perturbation to 1.0 micron

    fits.writeto(
        outpath/"M3_bend_grid.fits.gz",
        m3_grid,
        overwrite=True
    )
    fits.writeto(
        outpath/"M3_bend_coords.fits.gz",
        np.stack([m3x, m3y]),
        overwrite=True
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
