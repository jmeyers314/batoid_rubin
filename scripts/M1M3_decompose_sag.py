import pickle

from galsim.zernike import zernikeBasis, Zernike
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from tqdm import tqdm


class Gridder:
    def __init__(self, x, y, ngrid, R_outer):
        self.x = x
        self.y = y
        self.ngrid = ngrid
        self.R_outer = R_outer

        self.r = np.hypot(self.x, self.y)
        self.rmin, self.rmax = np.min(self.r), np.max(self.r)
        th = np.linspace(0, 2*np.pi, 100)
        self.points = np.vstack([
            np.hstack([x, (self.rmin-0.1)*np.cos(th), (self.rmax+0.1)*np.cos(th)]),
            np.hstack([y, (self.rmin-0.1)*np.sin(th), (self.rmax+0.1)*np.sin(th)]),
        ]).T

        self.x_grid = np.arange(-2, ngrid-2) * R_outer/(ngrid-5) - R_outer/2
        self.dx = (self.x_grid[-1] - self.x_grid[0])/(ngrid-1)
        self.xx, self.yy = np.meshgrid(self.x_grid, self.x_grid)
        self.rr = np.hypot(self.xx, self.yy)

    def grid(self, z):
        ip = CloughTocher2DInterpolator(
            self.points, np.hstack([z, np.zeros(200)])
        )
        z_grid = ip(self.xx, self.yy)
        dd = self.dx / 10
        dzdx_grid = (ip(self.xx+dd, self.yy) - ip(self.xx-dd, self.yy))/(2*dd)
        dzdy_grid = (ip(self.xx, self.yy+dd) - ip(self.xx, self.yy-dd))/(2*dd)
        d2zdxy_grid = (  (ip(self.xx+dd, self.yy+dd) - ip(self.xx-dd, self.yy+dd))
                       - (ip(self.xx+dd, self.yy-dd) - ip(self.xx-dd, self.yy-dd))
                      )/(4*dd*dd)

        return z_grid, dzdx_grid, dzdy_grid, d2zdxy_grid


def main(args):
    M1_outer = 4.18
    M3_outer = 2.508
    if args.annular:
        M1_inner = 2.558
        M3_inner = 0.55
    else:
        M1_inner = 0.0
        M3_inner = 0.0

    with open(args.input, 'rb') as f:
        x, y, w1, w3, Udn3norm, Vdn3norm = pickle.load(f)
    nnode, nmode = Udn3norm.shape

    M1zk = np.zeros((nnode, args.jmax+1))
    M3zk = np.zeros((nnode, args.jmax+1))
    M1M3zk = np.zeros((nnode, args.jmax+1))

    if args.zk_simultaneous:
        basis = zernikeBasis(
            args.jmax, x, y, R_inner=M1_inner, R_outer=M1_outer
        )
        for imode in tqdm(range(nmode)):
            coefs, *_ = np.linalg.lstsq(basis.T, Udn3norm[:, imode], rcond=None)
            Udn3norm[:, imode] -= Zernike(
                coefs, R_inner=M1_inner, R_outer=M1_outer
            )(x, y)
            M1M3zk[imode] = coefs
    else:
        basis1 = zernikeBasis(
            args.jmax, x[w1], y[w1], R_inner=M1_inner, R_outer=M1_outer
        )
        basis3 = zernikeBasis(
            args.jmax, x[w3], y[w3], R_inner=M3_inner, R_outer=M3_outer
        )
        for imode in tqdm(range(nmode)):
            coefs1, *_ = np.linalg.lstsq(basis1.T, Udn3norm[w1, imode], rcond=None)
            Udn3norm[w1, imode] -= Zernike(
                coefs1, R_inner=M1_inner, R_outer=M1_outer
            )(x[w1], y[w1])
            M1zk[imode] = coefs1
            coefs3, *_ = np.linalg.lstsq(basis3.T, Udn3norm[w3, imode], rcond=None)
            Udn3norm[w3, imode] -= Zernike(
                coefs3, R_inner=M3_inner, R_outer=M3_outer
            )(x[w3], y[w3])
            M3zk[imode] = coefs3

    M1_z_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M1_dzdx_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M1_dzdy_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M1_d2zdxy_grid = np.empty((nmode, args.ngrid, args.ngrid))

    M3_z_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M3_dzdx_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M3_dzdy_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M3_d2zdxy_grid = np.empty((nmode, args.ngrid, args.ngrid))

    m1gridder = Gridder(x[w1], y[w1], args.ngrid, M1_outer)
    m3gridder = Gridder(x[w3], y[w3], args.ngrid, M3_outer)
    for imode in tqdm(range(nmode)):
        m1_result = m1gridder.grid(Udn3norm[w1, imode])
        M1_z_grid[imode] = m1_result[0]
        M1_dzdx_grid[imode] = m1_result[1]
        M1_dzdy_grid[imode] = m1_result[2]
        M1_d2zdxy_grid[imode] = m1_result[3]
        m3_result = m3gridder.grid(Udn3norm[w3, imode])
        M3_z_grid[imode] = m3_result[0]
        M3_dzdx_grid[imode] = m3_result[1]
        M3_dzdy_grid[imode] = m3_result[2]
        M3_d2zdxy_grid[imode] = m3_result[3]

    with open(args.output, 'wb') as f:
        pickle.dump(
            (M1zk, M3zk, M1M3zk,
             m1gridder.x_grid, M1_z_grid, M1_dzdx_grid, M1_dzdy_grid, M1_d2zdxy_grid,
             m3gridder.x_grid, M3_z_grid, M3_dzdx_grid, M3_dzdy_grid, M3_d2zdxy_grid),
            f
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M1M3_sag.pkl",
        help="Input pickle file.  Default: M1M3_sag.pkl"
    )
    parser.add_argument(
        "output",
        default="M1M3_decompose.pkl",
        help=
            "output file name  "
            "Default: M1M3_decompose.pkl",
        nargs='?'
    )
    parser.add_argument(
        "--zk_simultaneous",
        action="store_true",
        help=
            "Use simultaneous Zernike decomposition of M1M3 sag.  "
            "Default: False"
    )
    parser.add_argument(
        "--annular",
        action="store_true",
        help="Use annular Zernike decomposition.  Default: False"
    )
    parser.add_argument(
        "--jmax",
        type=int,
        default=28,
        help="Maximum Zernike index to use.  Default: 28"
    )
    parser.add_argument(
        "--ngrid",
        type=int,
        default=204,
        help="Number of grid points to use.  Default: 204"
    )
    args = parser.parse_args()
    main(args)
