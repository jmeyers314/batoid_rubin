from functools import lru_cache
import os

import astropy.io.fits as fits
from galsim.zernike import Zernike
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml


@lru_cache
def _fits_cache(datadir, fn):
    return fits.getdata(
        os.path.join(
            datadir,
            fn
        )
    )


def load_mirror(config, indir, i):
    zk = _fits_cache(indir, config['zk']['file'])[i]
    R_outer = config['zk']['R_outer']
    R_inner = config['zk']['R_inner']
    Z = Zernike(zk, R_outer=R_outer, R_inner=R_inner)

    x, y = _fits_cache(indir, config['grid']['coords'])
    xx, yy = np.meshgrid(x, y)
    zz, *_ = _fits_cache(indir, config['grid']['file'])
    zz = zz[i]

    return xx, yy, zz + Z(xx, yy)


def main(args):
    with open(os.path.join(args.indir, "bend.yaml")) as f:
        bend = yaml.safe_load(f)

    # M1M3
    fig, axes = plt.subplots(
        nrows=6, ncols=5, figsize=(8.5, 11)
    )
    for i, ax in enumerate(axes.flat):
        xx, yy, zz = load_mirror(bend['M1'], args.indir, i)
        rr = np.hypot(xx, yy)
        ww = (rr <= 4.18) & (rr >= 2.558)
        xx = xx[ww]
        yy = yy[ww]
        zz = zz[ww]
        ax.scatter(xx, yy, c=zz, s=0.1, cmap='seismic', vmin=-2e-6, vmax=2e-6)

        xx, yy, zz = load_mirror(bend['M3'], args.indir, i)
        rr = np.hypot(xx, yy)
        ww = (rr <= 2.508) & (rr >= 0.55)
        xx = xx[ww]
        yy = yy[ww]
        zz = zz[ww]
        ax.scatter(xx, yy, c=zz, s=0.1, cmap='seismic', vmin=-2e-6, vmax=2e-6)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.tight_layout()

    # M2
    fig2, axes2 = plt.subplots(
        nrows=6, ncols=5, figsize=(8.5, 11)
    )
    for i, ax in enumerate(axes2.flat):
        xx, yy, zz = load_mirror(bend['M2'], args.indir, i)
        rr = np.hypot(xx, yy)
        ww = (rr <= 1.71) & (rr >= 0.9)
        xx = xx[ww]
        yy = yy[ww]
        zz = zz[ww]
        ax.scatter(xx, yy, c=zz, s=0.1, cmap='seismic', vmin=-2e-6, vmax=2e-6)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'indir',
        type=str,
        default='output',
        help='Directory containing the decomposition input files'
    )
    args = parser.parse_args()
    main(args)