import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import Delaunay
from sklearn.neighbors import KNeighborsRegressor


def process_fits(raw_file: Path, proc_file: Path) -> None:
    """Process FITS with focal plane height measurements
    into an Astropy table.

    - Reads all BinTable HDUs whose EXTNAME starts with 'R'.
    - Normalizes z to zero median over science sensors within
    317 mm radius.
    - Removes CWFS offsets (+1.5 mm for SW1, -1.5 mm for SW0).
    """
    rows = []
    with fits.open(raw_file) as hdul:
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU):
                extname = hdu.header.get("EXTNAME", "")
                if extname.startswith("R"):
                    tab = Table(hdu.data)
                    # Normalize detector name like R40_S00, and convert WFS->SW
                    detector = (extname[:3] + "_" + extname[3:]).replace("WFS", "SW")
                    for x, y, z_mod, z_meas in zip(
                        tab["X_CCS"],
                        tab["Y_CCS"],
                        tab["Z_CCS_MODEL"],
                        tab["Z_CCS_MEASURED"],
                    ):
                        rows.append([x, y, detector, z_mod, z_meas])

    data = Table(
        rows=rows,
        names=["x_ccs", "y_ccs", "detector", "z_model_raw", "z_measured_raw"],
        units=2 * ["mm"] + [None] + 2 * ["mm"],
    )

    # Set zero-point to median of science sensors within 317mm of center
    r = np.sqrt(data["x_ccs"] ** 2 + data["y_ccs"] ** 2)
    is_science = np.array(
        [(("S" in d) and ("W" not in d)) for d in data["detector"]], dtype=bool
    )
    mask = (r < 317.0) & is_science

    z_model_zeropt = np.median(data["z_model_raw"][mask])
    z_meas_zeropt = np.median(data["z_measured_raw"][mask])

    data["z_model"] = data["z_model_raw"] - z_model_zeropt
    data["z_measured"] = data["z_measured_raw"] - z_meas_zeropt

    # Remove CWFS offsets
    for i in range(len(data)):
        det = str(data[i]["detector"])
        if det.endswith("SW1"):
            data[i]["z_model"] += 1.5
            data[i]["z_measured"] += 1.5
        elif det.endswith("SW0"):
            data[i]["z_model"] -= 1.5
            data[i]["z_measured"] -= 1.5

    data.write(proc_file, format='fits')


def interpolate_map(
    proc_file: Path,
    interp_file: Path,
    map_type: str = "measured",
    interp_grid: np.ndarray | None = None,
) -> None:
    """Interpolate focal plane height map onto a 
    regular grid and save (x,y,z) table.
    """
    if not proc_file.exists():
        raise FileNotFoundError(f"{proc_file} not found. Run --process first.")

    table = Table.read(proc_file)
    # Create interpolation grid across the focal plane
    if interp_grid is None:
        interp_grid = np.linspace(-320.0, +320.0, 1000)

    x_grid, y_grid = np.meshgrid(interp_grid, interp_grid)
    x = x_grid.ravel()
    y = y_grid.ravel()
    z = np.zeros_like(x, dtype=float)

    # Interpolate per-chip to avoid smearing across gaps
    for detector in np.unique(table["detector"]):
        chip = table[table["detector"] == detector]
        xmin, xmax = chip["x_ccs"].min(), chip["x_ccs"].max()
        ymin, ymax = chip["y_ccs"].min(), chip["y_ccs"].max()

        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if not np.any(mask):
            continue

        pts = np.column_stack((chip["x_ccs"], chip["y_ccs"]))
        vals = chip[f"z_{map_type}"]

        knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
        knn.fit(pts, vals)
        z[mask] = knn.predict(np.column_stack((x, y))[mask])

    # Mask outside the overall convex hull
    hull = Delaunay(np.column_stack([table["x_ccs"], table["y_ccs"]]))
    inside = hull.find_simplex(np.column_stack([x, y])) >= 0
    z[~inside] = np.nan

    out = Table([x, y, z], names=["x", "y", "z"], units=3 * ["mm"])
    print(f"[WRITE] {interp_file}")
    out.write(interp_file, format='fits')


def main():
    p = argparse.ArgumentParser(
        description="Process and interpolate LSSTCam focal plane height maps."
    )
    p.add_argument(
        "--raw_file",
        default="LSST_FP_cold_b_measurement_4col_bysurface.fits",
        help="Input FITS from camera team.",
    )
    p.add_argument(
        "--proc_file",
        default="LsstCam_focal_plane_heights.fits.gz",
        help="Output processed table (FITS).",
    )
    p.add_argument(
        "--interp_file",
        default="LsstCam_focal_plane_heights_interpolated.fits.gz",
        help="Output interpolated (x,y,z) table (FITS).",
    )
    p.add_argument("--process", action="store_true", help="Run processing step.")
    p.add_argument("--interpolate", action="store_true", help="Run interpolation step.")
    p.add_argument(
        "--map_type",
        choices=["measured", "model"],
        default="measured",
        help="Which z_* column to interpolate.",
    )
    p.add_argument(
        "--grid_min", type=float, default=-320.0, help="Interpolation grid min (mm)."
    )
    p.add_argument(
        "--grid_max", type=float, default=+320.0, help="Interpolation grid max (mm)."
    )
    p.add_argument(
        "--grid_n", type=int, default=1000, help="Interpolation grid size per axis."
    )
    args = p.parse_args()

    raw_file = Path(args.raw_file)
    proc_file = Path(args.proc_file)
    interp_file = Path(args.interp_file)
    grid = np.linspace(args.grid_min, args.grid_max, args.grid_n)

    # Default behavior: run both steps if neither flag is given
    do_process = args.process or (not args.process and not args.interpolate)
    do_interpolate = args.interpolate or (not args.process and not args.interpolate)

    if do_process:
        process_fits(raw_file, proc_file)

    if do_interpolate:
        interpolate_map(proc_file, interp_file, map_type=args.map_type, interp_grid=grid)


if __name__ == "__main__":
    main()
