from pathlib import Path
from functools import lru_cache

import astropy.io.fits as fits
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


def _node_to_grid(nodex, nodey, nodez, grid_coords):
    """Convert FEA nodes positions into grid of z displacements,
    first derivatives, and mixed 2nd derivative.

    Parameters
    ----------
    nodex, nodey, nodez : ndarray (M, )
        Positions of nodes
    grid_coords : ndarray (2, N)
        Output grid positions in x and y

    Returns
    -------
    grid : ndarray (4, N, N)
        1st slice is interpolated z-position.
        2nd slice is interpolated dz/dx
        3rd slice is interpolated dz/dy
        4th slice is interpolated d2z/dxdy
    """
    interp = CloughTocher2DInterpolator(
        np.array([nodex, nodey]).T,
        nodez,
        fill_value=0.0
    )

    x, y = grid_coords
    nx = len(x)
    ny = len(y)
    out = np.zeros((4, ny, nx))
    # Approximate derivatives with finite differences.  Make the finite
    # difference spacing equal to 1/10th the grid spacing.
    dx = np.mean(np.diff(x))*1e-1
    dy = np.mean(np.diff(y))*1e-1
    x, y = np.meshgrid(x, y)
    out[0] = interp(x, y)
    out[1] = (interp(x+dx, y) - interp(x-dx, y))/(2*dx)
    out[2] = (interp(x, y+dy) - interp(x, y-dy))/(2*dy)
    out[3] = (
        interp(x+dx, y+dy) -
        interp(x-dx, y+dy) -
        interp(x+dx, y-dy) +
        interp(x-dx, y-dy)
    )/(4*dx*dy)

    # Zero out the central hole
    r = np.hypot(x, y)
    rmin = np.min(np.hypot(nodex, nodey))
    w = r < rmin
    out[:, w] = 0.0

    return out


@lru_cache(maxsize=100)
def _fits_cache(datadir, fn):
    """Cache loading fits file data table

    Parameters
    ----------
    datadir : str
        Directory containing the fits file
    fn : string
        File name from datadir to load and cache

    Returns
    -------
    out : ndarray
        Loaded data.
    """
    return fits.getdata(Path(datadir) / fn)
