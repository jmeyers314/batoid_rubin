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


def attach_attr(**kwargs):
    """Decorator for attaching attributes to functions
    """
    def inner(f):
        for k, v in kwargs.items():
            setattr(f, k, v)
        return f
    return inner

def read_h5_map(fileset, dataset = '/dataset'):
    '''
    The method takes a list of h5 files, get the images, and average them to get a combined image array.
    
    Parameters:
    fileset: 
        a list of h5 files
    dataset:
        the dataset name in the h5 files
    
    Returns:
    data: 
        the averaged image array
    centerRow, centerCol, pixelSize
    '''
    i = 0
    if len(fileset) == 0:
        print('Error: empty fileset')
        sys.exit()
    for filename in fileset:
        h5file = os.path.join(filename)
        f = h5py.File(h5file,'r')
        data0 = f[dataset]
        if 'date' in data0.attrs.keys():
            if len(data0.attrs['date']) == 1:
                timeStamp = data0.attrs['date'][0].decode('ascii')
            else:
                timeStamp = data0.attrs['date'].decode('ascii')
        else:
            timeStamp = 'date not in h5 file.'
        if i==0:
            centerRow = data0.attrs['centerRow']
            centerCol = data0.attrs['centerCol']
            pixelSize = data0.attrs['pixelSize']
            data = data0[:]
        else:
            data += data0[:]
        f.close()
        filenameShort = filename

        print('%s: %s '%(filenameShort, timeStamp))
        i+=1
    data /= i
    data = np.rot90(data, 1) # so that we can use imshow(data, origin='lower')
    return data, centerRow, centerCol, pixelSize

def load_and_clip_m2_surface(file_path):
    data = scipy.io.loadmat(file_path)

    s = data['s']
    m2_data = s['z'][0][0] * 1e-3  # Convert from mm to um

    valid_mask = np.isfinite(m2_data)
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    row_min, row_max = np.where(valid_rows)[0][[0, -1]]
    col_min, col_max = np.where(valid_cols)[0][[0, -1]]
    m2_clipped = m2_data[row_min:row_max+1, col_min:col_max+1]

    return m2_clipped