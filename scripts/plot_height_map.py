import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import astropy.io.fits as fits
import numpy as np
from lsst.obs.lsst import LsstCam


def transform_name(name):
    name = name[:3]+"_"+name[3:]
    name = name.replace("GS", "SG").replace("WFS", "SW")
    if "SG" in name:
        if "SG0" in name:
            name = name.replace("SG0", "SG1")
        else:
            name = name.replace("SG1", "SG0")
    return name


def clip_columns(data):
    """
    Removes first and last ~4% of columns

    Data consists roughly of 50 columns.  The first and last 4% are frequently
    unreliable on E2V chips so we remove them.  The x-coordinates are not precisely
    gridded though - only approximately - so we use a simple range fraction clipping
    algorithm.
    """
    x = data["X_CCS"]
    xrange = np.nanquantile(x, [0, 1])
    xspan = np.ptp(xrange)
    w = (x > xrange[0] + xspan/25) & (x < xrange[1] - xspan/25)
    # Clip out leading/trailing 4% of columns
    return data[w]


raw_hdul = fits.open("LSST_FP_cold_b_measurement_4col_bysurface.fits")
processed_hdul = fits.open("ccd_height_map.fits.gz")
camera = LsstCam().getCamera()


def main(use_edcs=False):
    raw_data = raw_hdul[-1].data
    x = raw_data["X_CCS"]
    y = raw_data["Y_CCS"]
    z = raw_data["Z_CCS_MEASURED"]
    z[z > 1] -= 1.5
    z[z < -1] += 1.5

    raw_rms = np.std(z)
    print(f"Raw height data statistics:")
    print(f"  RMS: {raw_rms*1000:.3f} µm")
    print(f"  Mean: {np.mean(z)*1000:.3f} µm")
    print(f"  Min: {np.min(z)*1000:.3f} µm")
    print(f"  Max: {np.max(z)*1000:.3f} µm")
    print()

    vmin, vmax = -0.01, 0.01
    kwargs = dict(cmap="bwr", vmin=vmin, vmax=vmax, s=0.5)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), constrained_layout=True)
    if use_edcs:
        sc = ax.scatter(y, x, c=z, **kwargs)
    else:
        sc = ax.scatter(x, y, c=z, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-325, 325)
    ax.set_ylim(-325, 325)
    ax.set_aspect("equal")
    circle = Circle((0, 0), 317, edgecolor="black", facecolor="none", lw=1)
    ax.add_patch(circle)
    ax.set_title(f"Raw Height Data, RMS = {raw_rms*1000:.3f} µm", fontsize=16)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (mm)', fontsize=14)
    fig.savefig("raw_height_data.png", dpi=300)

    # CCD by CCD cutting out E2V columns
    vmin, vmax = -0.01, 0.01
    kwargs = dict(cmap="bwr", vmin=vmin, vmax=vmax, s=0.5)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), constrained_layout=True)
    all_z_clipped = []
    for hdu in raw_hdul[1:-1]:
        detname = transform_name(hdu.header["EXTNAME"])
        det = camera[detname]
        data = hdu.data
        if det.getPhysicalType() == "E2V":
            data = clip_columns(data)
        x = data["X_CCS"]
        y = data["Y_CCS"]
        z = data["Z_CCS_MEASURED"]
        z[z > 1] -= 1.5
        z[z < -1] += 1.5
        all_z_clipped.extend(z)
        if use_edcs:
            ax.scatter(y, x, c=z, **kwargs)
        else:
            ax.scatter(x, y, c=z, **kwargs)

    all_z_clipped = np.array(all_z_clipped)
    clipped_rms = np.std(all_z_clipped)
    print(f"Column-clipped height data statistics:")
    print(f"  RMS: {clipped_rms*1000:.3f} µm")
    print(f"  Mean: {np.mean(all_z_clipped)*1000:.3f} µm")
    print(f"  Min: {np.min(all_z_clipped)*1000:.3f} µm")
    print(f"  Max: {np.max(all_z_clipped)*1000:.3f} µm")
    print()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-325, 325)
    ax.set_ylim(-325, 325)
    ax.set_aspect("equal")
    circle = Circle((0, 0), 317, edgecolor="black", facecolor="none", lw=1)
    ax.add_patch(circle)
    ax.set_title(f"Column-Clipped Height Data, RMS = {clipped_rms*1000:.3f} µm", fontsize=16)
    # Get the last scatter plot for colorbar
    sc = ax.collections[-1]
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (mm)', fontsize=14)
    fig.savefig("column_clipped_height_data.png", dpi=300)

    vmin, vmax = -0.01, 0.01
    kwargs = dict(cmap="bwr", vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), constrained_layout=True)
    all_processed_z = []
    for hdu in processed_hdul:
        data = hdu.data.copy()
        if not np.isfinite(data).all():
            import ipdb; ipdb.set_trace()
        hdr = hdu.header
        xmin = hdr["XMIN"]
        xmax = hdr["XMAX"]
        ymin = hdr["YMIN"]
        ymax = hdr["YMAX"]
        x = np.linspace(xmin, xmax, data.shape[1] + 1)
        y = np.linspace(ymin, ymax, data.shape[0] + 1)
        data[data > 1] -= 1.5
        data[data < -1] += 1.5
        all_processed_z.extend(data.ravel())
        if use_edcs:
            mesh = ax.pcolormesh(y, x, data, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, data, **kwargs)

    all_processed_z = np.array(all_processed_z)
    processed_rms = np.std(all_processed_z)
    print(f"Processed height data statistics:")
    print(f"  RMS: {processed_rms*1000:.3f} µm")
    print(f"  Mean: {np.mean(all_processed_z)*1000:.3f} µm")
    print(f"  Min: {np.min(all_processed_z)*1000:.3f} µm")
    print(f"  Max: {np.max(all_processed_z)*1000:.3f} µm")
    print()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-325, 325)
    ax.set_ylim(-325, 325)
    ax.set_aspect("equal")
    circle = Circle((0, 0), 317, edgecolor="black", facecolor="none", lw=1)
    ax.add_patch(circle)
    ax.set_title(f"Processed Height Data, RMS = {processed_rms*1000:.3f} µm", fontsize=16)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (mm)', fontsize=14)
    fig.savefig("processed_height_data.png", dpi=300)

    # Height residuals plot - difference between raw measurements and processed interpolation
    vmin, vmax = -0.002, 0.002
    kwargs = dict(cmap="bwr", vmin=vmin, vmax=vmax, s=0.5)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), constrained_layout=True)

    # For each raw data point, compute the residual from the processed height map
    residuals_x = []
    residuals_y = []
    residuals_z = []

    for hdu in raw_hdul[1:-1]:
        detname = transform_name(hdu.header["EXTNAME"])
        det = camera[detname]
        data = hdu.data
        if det.getPhysicalType() == "E2V":
            data = clip_columns(data)

        x = data["X_CCS"]
        y = data["Y_CCS"]
        z = data["Z_CCS_MEASURED"]
        z[z > 1] -= 1.5
        z[z < -1] += 1.5

        # Find the corresponding processed HDU for this detector
        for proc_hdu in processed_hdul:
            if proc_hdu.header.get("EXTNAME") == detname:
                proc_data = proc_hdu.data
                proc_hdr = proc_hdu.header
                xmin = proc_hdr["XMIN"]
                xmax = proc_hdr["XMAX"]
                ymin = proc_hdr["YMIN"]
                ymax = proc_hdr["YMAX"]

                # Apply same corrections to processed data
                proc_data[proc_data > 1] -= 1.5
                proc_data[proc_data < -1] += 1.5

                # Interpolate processed data at raw measurement locations
                ny, nx = proc_data.shape
                x_indices = (x - xmin) / (xmax - xmin) * nx
                y_indices = (y - ymin) / (ymax - ymin) * ny

                # Only keep points within bounds
                valid = (x_indices >= 0) & (x_indices < nx) & (y_indices >= 0) & (y_indices < ny)

                for i in range(len(x)):
                    if valid[i]:
                        xi = int(x_indices[i])
                        yi = int(y_indices[i])
                        # Ensure indices are within array bounds
                        if 0 <= yi < ny and 0 <= xi < nx:
                            interpolated_z = proc_data[yi, xi]
                            residual = z[i] - interpolated_z
                            residuals_x.append(x[i])
                            residuals_y.append(y[i])
                            residuals_z.append(residual)
                break

    residuals_x = np.array(residuals_x)
    residuals_y = np.array(residuals_y)
    residuals_z = np.array(residuals_z)

    if use_edcs:
        sc = ax.scatter(residuals_y, residuals_x, c=residuals_z, **kwargs)
    else:
        sc = ax.scatter(residuals_x, residuals_y, c=residuals_z, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-325, 325)
    ax.set_ylim(-325, 325)
    ax.set_aspect("equal")
    circle = Circle((0, 0), 317, edgecolor="black", facecolor="none", lw=1)
    ax.add_patch(circle)
    residuals_rms = np.std(residuals_z)
    ax.set_title(f"Height Residuals (Raw - Processed), RMS = {residuals_rms*1000:.3f} µm", fontsize=16)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual (mm)', fontsize=14)
    fig.savefig("height_residuals.png", dpi=300)

    print(f"Height residuals statistics:")
    print(f"  RMS: {residuals_rms*1000:.3f} µm")
    print(f"  Mean: {np.mean(residuals_z)*1000:.3f} µm")
    print(f"  Min: {np.min(residuals_z)*1000:.3f} µm")
    print(f"  Max: {np.max(residuals_z)*1000:.3f} µm")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--EDCS", action="store_true", help="Use EDCS coordinates for plotting")
    args = parser.parse_args()
    main(args.EDCS)