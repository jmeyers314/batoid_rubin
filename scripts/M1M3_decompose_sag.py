import pickle
import batoid

from galsim.zernike import zernikeBasis, Zernike
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import CloughTocher2DInterpolator
from tqdm import tqdm


class Gridder:
    def __init__(self, x, y, ngrid, R_outer, interp='ct'):
        self.x = x
        self.y = y
        self.ngrid = ngrid
        self.R_outer = R_outer
        self.interp = interp

        self.r = np.hypot(self.x, self.y)
        self.rmin, self.rmax = np.min(self.r), np.max(self.r)
        th = np.linspace(0, 2*np.pi, 100)
        self.points = np.vstack([
            np.hstack([x, (self.rmin-0.1)*np.cos(th), (self.rmax+0.1)*np.cos(th)]),
            np.hstack([y, (self.rmin-0.1)*np.sin(th), (self.rmax+0.1)*np.sin(th)]),
        ]).T

        self.x_grid = np.arange(-2, ngrid-2) * 2*R_outer/(ngrid-5) - R_outer
        self.dx = (self.x_grid[-1] - self.x_grid[0])/(ngrid-1)
        self.xx, self.yy = np.meshgrid(self.x_grid, self.x_grid)
        self.rr = np.hypot(self.xx, self.yy)

    def grid(self, z):
        if self.interp == 'ct':
            ip = CloughTocher2DInterpolator(
                self.points, np.hstack([z, np.zeros(200)])
            )
        else:
            basis = zernikeBasis(
                171, self.points[:-200, 0], self.points[:-200, 1],
                R_outer=self.R_outer
            )
            coefs, *_ = np.linalg.lstsq(basis.T, z, rcond=None)
            ip = Zernike(
                coefs, R_outer=self.R_outer
            )
        z_grid = ip(self.xx, self.yy)
        dd = self.dx / 10
        dzdx_grid = (
            ip(self.xx+dd, self.yy) -
            ip(self.xx-dd, self.yy)
        )/(2*dd)
        dzdy_grid = (
            ip(self.xx, self.yy+dd) -
            ip(self.xx, self.yy-dd)
        )/(2*dd)
        d2zdxy_grid = (
            (ip(self.xx+dd, self.yy+dd) -
             ip(self.xx-dd, self.yy+dd)) -
            (ip(self.xx+dd, self.yy-dd) -
             ip(self.xx-dd, self.yy-dd))
        )/(4*dd*dd)

        for arr in z_grid, dzdx_grid, dzdy_grid, d2zdxy_grid:
            arr[self.rr > (self.rmax+0.05)] = 0
            arr[self.rr < (self.rmin-0.05)] = 0

        return z_grid, dzdx_grid, dzdy_grid, d2zdxy_grid


def main(args):
    M1_outer = 4.18
    M3_outer = 2.508
    if args.circular:
        M1_inner = 0.0
        M3_inner = 0.0
    else:
        M1_inner = 2.558
        M3_inner = 0.55

    if args.input.endswith('.pkl'):
        with open(args.input, 'rb') as f:
            x, y, w1, w3, Udn3norm, Vdn3norm = pickle.load(f)
    elif args.input.endswith('.mat'):
        data = loadmat(args.input)
        x = data['x'][:, 0]
        y = data['y'][:, 0]
        w1 = data['annulus'][:, 0] == 1
        w3 = data['annulus'][:, 0] == 3
        Udn3norm = data['Udn3norm']
        Vdn3norm = data['Vdn3norm']
    r = np.hypot(x, y)
    # Points in common between M1 and M3
    w13 = (r > 2.53) & (r < 2.54)
    w1 &= ~w13  # Points unique to M1

    nnode, nmode = Udn3norm.shape

    M1zk = np.zeros((nmode, args.jmax+1))
    M3zk = np.zeros((nmode, args.jmax+1))
    M1M3zk = np.zeros((nmode, args.jmax+1))

    zk_eval1 = np.zeros((nmode, len(x)))
    zk_eval3 = np.zeros((nmode, len(x)))
    if args.zk_simultaneous:
        basis = zernikeBasis(
            args.jmax, x, y, R_inner=M1_inner, R_outer=M1_outer
        )
        for imode in tqdm(range(nmode)):
            coefs, *_ = np.linalg.lstsq(basis.T, Udn3norm[:, imode], rcond=None)
            M1M3zk[imode] = coefs
            zk_eval1[imode] = Zernike(
                coefs, R_inner=M1_inner, R_outer=M1_outer
            )(x, y)
        zk_eval3[:] = zk_eval1
    else:
        basis1 = zernikeBasis(
            args.jmax, x[w1], y[w1], R_inner=M1_inner, R_outer=M1_outer
        )
        basis3 = zernikeBasis(
            args.jmax, x[w3], y[w3], R_inner=M3_inner, R_outer=M3_outer
        )
        for imode in tqdm(range(nmode)):
            coefs1, *_ = np.linalg.lstsq(basis1.T, Udn3norm[w1, imode], rcond=None)
            M1zk[imode] = coefs1
            zk_eval1[imode] = Zernike(
                coefs1, R_inner=M1_inner, R_outer=M1_outer
            )(x, y)

            coefs3, *_ = np.linalg.lstsq(basis3.T, Udn3norm[w3, imode], rcond=None)
            M3zk[imode] = coefs3
            zk_eval3[imode] = Zernike(
                coefs3, R_inner=M3_inner, R_outer=M3_outer
            )(x, y)

    M1_z_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M1_dzdx_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M1_dzdy_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M1_d2zdxy_grid = np.empty((nmode, args.ngrid, args.ngrid))

    M3_z_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M3_dzdx_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M3_dzdy_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M3_d2zdxy_grid = np.empty((nmode, args.ngrid, args.ngrid))

    if args.share_m1m3_interface:
        telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        # Use M3 normal instead of M1 normal for shared interface points
        tmp = np.array(Udn3norm)
        tmp[w13] *= telescope['M1'].surface.normal(x[w13], y[w13])[:, 2][:, None]
        tmp[w13] /= telescope['M3'].surface.normal(x[w13], y[w13])[:, 2][:, None]
        z3 = tmp[w3|w13].T - zk_eval3[:, w3|w13]
        x3 = x[w3|w13]
        y3 = y[w3|w13]

        x1 = x[w1|w13]
        y1 = y[w1|w13]
        z1 = Udn3norm[w1|w13].T - zk_eval1[:, w1|w13]
    else:
        z3 = Udn3norm[w3].T - zk_eval3[:, w3]
        x3 = x[w3]
        y3 = y[w3]

        x1 = x[w1]
        y1 = y[w1]
        z1 = Udn3norm[w1].T - zk_eval1[:, w1]

    m1gridder = Gridder(x1, y1, args.ngrid, M1_outer, interp='z')
    m3gridder = Gridder(x3, y3, args.ngrid, M3_outer, interp='z')
    for imode in tqdm(range(nmode)):
        m1_result = m1gridder.grid(z1[imode])
        M1_z_grid[imode] = m1_result[0]
        M1_dzdx_grid[imode] = m1_result[1]
        M1_dzdy_grid[imode] = m1_result[2]
        M1_d2zdxy_grid[imode] = m1_result[3]

        m3_result = m3gridder.grid(z3[imode])
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
        help="Input pickle or mat file.  Default: M1M3_sag.pkl"
    )
    parser.add_argument(
        "output",
        default="M1M3_decomposition.pkl",
        help=
            "output file name  "
            "Default: M1M3_decomposition.pkl",
        nargs='?'
    )
    parser.add_argument(
        "--zk_simultaneous",
        action="store_true",
        help=
            "Use simultaneous (instead of independent) Zernike "
            "decomposition of M1 and M3 sag.  "
            "Default: False"
    )
    parser.add_argument(
        "--circular",
        action="store_true",
        help=
            "Use circular (instead of annular) Zernike decomposition.  "
            "Default: False"
    )
    parser.add_argument(
        "--share_m1m3_interface",
        action="store_true",
        help="Share FEA points at interface between M1 and M3 for grid determination."
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
