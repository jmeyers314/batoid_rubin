import os
from pathlib import Path
import argparse

import numpy as np
import astropy.io.fits as fits
import h5py
import scipy.io


def read_h5_map(path, filename, dataset='/dataset'):
    """Read an HDF5 map (µm) and return a rotated array in meters."""
    with h5py.File(Path(path) / filename, 'r') as f:
        data = np.rot90(f[dataset], 1)  # so imshow(..., origin='lower') works
    return data * 1e-6  # um → m


def load_and_clip_m2_surface(path, filename):
    """Load M2 surface from .mat (nm → m) and clip to finite valid rectangle."""
    data = scipy.io.loadmat(Path(path) / filename)
    m2_data = data['s']['z'][0][0] * 1e-9  # nm → m

    valid = np.isfinite(m2_data)
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    return m2_data[r0:r1 + 1, c0:c1 + 1]


def write_surface_to_fits(x, y, surface, outdir, name):
    """
    Write coordinate grid (X,Y) and surface arrays to FITS.
    x, y : 1D arrays (meters)
    surface : 2D array (meters), shape = (len(y), len(x))
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, Y = np.meshgrid(x, y)

    fits.writeto(outdir / f"{name}_figure_coords.fits.gz",
                 np.stack([X, Y]),
                 overwrite=True)

    fits.writeto(outdir / f"{name}_figure_surface.fits.gz",
                 surface.astype(np.float64),
                 overwrite=True)


def export_mirror_figure_errors(mirror_figure_dir, outdir,
                                m1_file='m1_figure_error.h5',
                                m3_file='m3_figure_error.h5',
                                m2_file='m2_figure_error.mat',
                                m1_extent=4.18, m1_n=986,
                                m3_extent=2.508, m3_n=1006,
                                m2_extent=1.71, m2_n=685,
                                h5_dataset='/dataset'):
    """
    Build grid coords from provided extents and write
    *_figure_coords.fits.gz and *_figure_surface.fits.gz for M1, M2, M3.
    """
    mirror_figure_dir = Path(mirror_figure_dir)

    # --- M1 ---
    x_m1 = np.linspace(-m1_extent, m1_extent, m1_n)
    y_m1 = np.linspace(-m1_extent, m1_extent, m1_n)
    m1s = read_h5_map(mirror_figure_dir, m1_file, dataset=h5_dataset)
    m1s /= 2 # M1 surface error is 2x the figure error 
    m1s = np.nan_to_num(m1s, nan=0.0)
    write_surface_to_fits(x_m1, y_m1, m1s, outdir, "M1")

    # --- M3 ---
    x_m3 = np.linspace(-m3_extent, m3_extent, m3_n)
    y_m3 = np.linspace(-m3_extent, m3_extent, m3_n)
    m3s = read_h5_map(mirror_figure_dir, m3_file, dataset=h5_dataset)
    m3s /= 2 # M3 surface error is 2x the figure error 
    m3s = np.nan_to_num(m3s, nan=0.0)
    write_surface_to_fits(x_m3, y_m3, m3s, outdir, "M3")

    # --- M2 ---
    x_m2 = np.linspace(-m2_extent, m2_extent, m2_n)
    y_m2 = np.linspace(-m2_extent, m2_extent, m2_n)
    m2s = load_and_clip_m2_surface(mirror_figure_dir, m2_file)
    m2s = np.nan_to_num(m2s, nan=0.0)
    m2s /= 2 # M2 surface error is 2x the figure error 
    write_surface_to_fits(x_m2, y_m2, m2s, outdir, "M2")


def main():
    p = argparse.ArgumentParser(
        description="Export M1/M2/M3 mirror figure errors to FITS (coords + surface)."
    )
    p.add_argument("--mirror_figure_dir", type=str, required=True,
                   help="Directory containing m1_figure_error.h5, m3_figure_error.h5, m2_figure_error.mat")
    p.add_argument("--outdir", type=str, default="batoid_mirror_figures",
                   help="Output directory (default: batoid_mirror_figures)")
    p.add_argument("--m1_file", type=str, default="m1_figure_error.h5")
    p.add_argument("--m3_file", type=str, default="m3_figure_error.h5")
    p.add_argument("--m2_file", type=str, default="m2_figure_error.mat")

    # Grid params (override if your extents/resolutions differ)
    p.add_argument("--m1_extent", type=float, default=4.18, help="Half-size in meters for M1 grid (±extent)")
    p.add_argument("--m1_n", type=int, default=986, help="M1 grid size (per axis)")

    p.add_argument("--m3_extent", type=float, default=2.508, help="Half-size in meters for M3 grid (±extent)")
    p.add_argument("--m3_n", type=int, default=1006, help="M3 grid size (per axis)")

    p.add_argument("--m2_extent", type=float, default=1.71, help="Half-size in meters for M2 grid (±extent)")
    p.add_argument("--m2_n", type=int, default=685, help="M2 grid size (per axis)")

    args = p.parse_args()

    export_mirror_figure_errors(
        mirror_figure_dir=args.mirror_figure_dir,
        outdir=args.outdir,
        m1_file=args.m1_file,
        m3_file=args.m3_file,
        m2_file=args.m2_file,
        m1_extent=args.m1_extent, m1_n=args.m1_n,
        m3_extent=args.m3_extent, m3_n=args.m3_n,
        m2_extent=args.m2_extent, m2_n=args.m2_n,
    )
    print(f"Wrote FITS files to: {args.outdir}")


if __name__ == "__main__":
    main()