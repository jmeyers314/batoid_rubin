import astropy.io.fits as fits
from galsim.zernike import zernikeBasis, Zernike
from lsst.obs.lsst import LsstCam
from lsst.afw.cameraGeom import FOCAL_PLANE
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from tqdm import tqdm


def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


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


def get_domain(det):
    corners = det.getCorners(FOCAL_PLANE)
    corners = [(y, x) for x, y in corners]  # FLIP DVCS -> CCS
    xmin, xmax = np.nanquantile([c[0] for c in corners], [0, 1])
    ymin, ymax = np.nanquantile([c[1] for c in corners], [0, 1])
    xx = np.linspace(xmin, xmax, 100)
    yy = np.linspace(ymin, ymax, 100)
    xx, yy = np.meshgrid(xx, yy)
    xmid = 0.5*(xmin + xmax)
    ymid = 0.5*(ymin + ymax)
    xscale = xmax - xmin
    yscale = ymax - ymin
    return {
        "corners": corners,
        "xmin": xmin, # FOCAL_PLANE CCS
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "xx": xx, # FOCAL_PLANE CCS
        "yy": yy,
        "xmid": xmid, # FOCAL_PLANE CCS
        "ymid": ymid,
        "xscale": xscale,
        "yscale": yscale,
        "xx_scaled": (xx - xmid) / xscale,  # ~CCD coords
        "yy_scaled": (yy - ymid) / yscale
    }


def load_domains(camera):
    domains = {}
    for det in camera:
        domains[det.getName()] = get_domain(det)
    return domains


def get_table(hdul, domains):
    tables = []
    for hdu in hdul[1:-1]:
        table = Table(hdu.data)
        detname = transform_name(hdu.header["EXTNAME"])
        table["detname"] = detname

        domain = domains[detname]
        table["X_CCD"] = (table["X_CCS"] - domain["xmid"]) / domain["xscale"]
        table["Y_CCD"] = (table["Y_CCS"] - domain["ymid"]) / domain["yscale"]

        tables.append(table)
    return vstack(tables)


def get_pca(camera, table, domains, jmax, npca=20, physical_type=None, clip_edges=False):
    if physical_type is None:
        raise ValueError("physical_type must be specified")
    zk_fits = []
    for detname in np.unique(table["detname"]):
        det = camera[detname]
        if det.getPhysicalType() != physical_type:
            continue
        data = table[table["detname"] == detname]
        # For building the PCA, throw away sensors with significant missing data
        # These are the ones near the edge of the focal plane.
        if len(data["Z_CCS_MEASURED"]) < 2447:
            continue

        if clip_edges:
            data = clip_columns(data)

        # Interpolation domain
        domain = domains[detname]

        basis = zernikeBasis(jmax, data["X_CCD"], data["Y_CCD"])
        zk_coef, *_ = np.linalg.lstsq(basis.T, data["Z_CCS_MEASURED"])
        zk_val = Zernike(zk_coef)(domain["xx_scaled"], domain["yy_scaled"])
        # Clip to original range
        zk_val = np.clip(zk_val, *np.nanquantile(data["Z_CCS_MEASURED"], [0, 1]))
        # Gridded Zernike interpolation
        zk_fits.append(zk_val)
    zk_fits = np.array(zk_fits)

    zk_fits_mean = np.mean(zk_fits, axis=0)
    zk_fits -= zk_fits_mean
    u, s, vt = np.linalg.svd(zk_fits.reshape(len(zk_fits), -1), full_matrices=False)
    vt = vt.reshape(vt.shape[0], zk_fits[0].shape[0], zk_fits[0].shape[1])

    # Fudge the range a tiny bit to avoid floating point issues with the interpolation.
    eps = 1e-15
    ip_x = np.linspace(-0.5 - eps, 0.5 + eps, 100)
    ip_y = np.linspace(-0.5 - eps, 0.5 + eps, 100)
    ip_xx, ip_yy = np.meshgrid(ip_x, ip_y)

    ip_mean = CloughTocher2DInterpolator(
        np.array([ip_xx.flatten(), ip_yy.flatten()]).T, zk_fits_mean.flatten()
    )
    ip_vt = []
    for i in range(npca):
        ip = CloughTocher2DInterpolator(
            np.array([ip_xx.flatten(), ip_yy.flatten()]).T, vt[i].flatten()
        )
        ip_vt.append(ip)

    return ip_mean, ip_vt


def main(input_file, jmax, npca, output_file):
    hdul = fits.open(input_file)
    camera = LsstCam.getCamera()
    domains = load_domains(camera)
    table = get_table(hdul, domains)

    e2v_mean, e2v_vt = get_pca(camera, table, domains, jmax, npca, physical_type="E2V", clip_edges=True)
    itl_mean, itl_vt = get_pca(camera, table, domains, jmax, npca, physical_type="ITL", clip_edges=False)

    output_hdul = fits.HDUList()

    for det in tqdm(camera):
        detname = det.getName()
        detnum = det.getId()
        domain = domains[det.getName()]
        subtable = table[table["detname"] == detname]
        if det.getPhysicalType() == "ITL_WF":
            # Special, not enough to PCA, but don't really need it.
            basis = zernikeBasis(jmax, subtable["X_CCD"], subtable["Y_CCD"])
            z_data = subtable["Z_CCS_MEASURED"]
            z_data[z_data > 1] -= 1.5
            z_data[z_data < -1] += 1.5
            coef, *_ = np.linalg.lstsq(basis.T, z_data, rcond=None)
            z_grid = Zernike(coef)(domain["xx_scaled"], domain["yy_scaled"])
        else:
            if det.getPhysicalType() == "E2V":
                ip_mean, ip_vt = e2v_mean, e2v_vt
                subtable = clip_columns(subtable)
            elif det.getPhysicalType() in ["ITL", "ITL_G"]:
                ip_mean, ip_vt = itl_mean, itl_vt
            else:
                raise ValueError(f"Unknown physical type: {det.getPhysicalType()}")
            # Project onto first npca components
            x = subtable["X_CCD"]
            y = subtable["Y_CCD"]
            z = subtable["Z_CCS_MEASURED"]
            xx = domain["xx_scaled"]
            yy = domain["yy_scaled"]

            # ITL_G sensors need to be rotated
            if det.getId() in [197, 202]:
                pass
            elif det.getId() in [190, 193]:
                x, y = -x, -y
                xx, yy = -xx, -yy
            elif det.getId() in [189, 198]:
                x, y = -y, x
                xx, yy = -yy, xx
            elif det.getId() in [194, 201]:
                x, y = y, -x
                xx, yy = yy, -xx

            z_centered = z - ip_mean(x, y)
            basis = np.array([ip(x, y) for ip in ip_vt[:npca]]).T
            coef, *_ = np.linalg.lstsq(basis, z_centered, rcond=None)
            z_grid = ip_mean(xx, yy) + sum(c*ip(xx, yy) for c, ip in zip(coef, ip_vt))

        # Let's write out a zk fit for each sensor as well and embed it in the header.
        xx, yy = domain["xx_scaled"], domain["yy_scaled"]
        basis = zernikeBasis(jmax, xx.ravel(), yy.ravel())
        coef, *_ = np.linalg.lstsq(basis.T, z_grid.ravel(), rcond=None)
        rssr = np.std(z_grid - Zernike(coef)(xx, yy))

        if not np.isfinite(z_grid).all():
            import ipdb; ipdb.set_trace()

        hdu = fits.ImageHDU(z_grid.astype(np.float32), name=detname)
        hdu.header["BUNIT"] = "mm"
        hdu.header["DETNAME"] = detname
        hdu.header["DETTYPE"] = det.getPhysicalType()
        hdu.header["DETNUM"] = detnum
        hdu.header["XMIN"] = (domain["xmin"], "[mm]")
        hdu.header["XMAX"] = (domain["xmax"], "[mm]")
        hdu.header["YMIN"] = (domain["ymin"], "[mm]")
        hdu.header["YMAX"] = (domain["ymax"], "[mm]")
        hdu.header["XMID"] = (domain["xmid"], "[mm]")
        hdu.header["YMID"] = (domain["ymid"], "[mm]")
        for j, c in enumerate(coef[1:], start=1):
            hdu.header[f"ZK{j:02d}"] = c
        hdu.header["ZK_RSSR"] = rssr
        output_hdul.append(hdu)

    output_hdul.writeto(output_file, overwrite=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="LSST_FP_cold_b_measurement_4col_bysurface.fits", help="Input FITS file")
    parser.add_argument("--output", default="ccd_height_map.fits.gz", help="Output FITS file")
    parser.add_argument("--jmax", type=int, default=36, help="Maximum Zernike order")
    parser.add_argument("--npca", type=int, default=20, help="Number of PCA components")
    args = parser.parse_args()

    main(args.input, args.jmax, args.npca, args.output)
