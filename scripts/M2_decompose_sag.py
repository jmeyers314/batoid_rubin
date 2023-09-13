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
    M2_outer = 1.71
    if args.circular:
        M2_inner = 0.0
    else:
        M2_inner = 0.9

    if args.input.endswith('.pkl'):
        with open(args.input, 'rb') as f:
            x, y, Udn3norm, Vdn3norm = pickle.load(f)
    elif args.input.endswith('.mat'):
        data = loadmat(args.input)
        x = data['x'][:, 0]
        y = data['y'][:, 0]
        Udn3norm = data['Udn3norm']
        Vdn3norm = data['Vdn3norm']
    nnode, nmode = Udn3norm.shape

    M2zk = np.zeros((nmode, args.jmax+1))

    basis = zernikeBasis(
        args.jmax, x, y, R_inner=M2_inner, R_outer=M2_outer
    )
    zk_eval2 = np.zeros((nmode, len(x)))
    for imode in tqdm(range(nmode)):
        coefs, *_ = np.linalg.lstsq(basis.T, Udn3norm[:, imode], rcond=None)
        M2zk[imode] = coefs
        zk_eval2[imode] = Zernike(
            coefs, R_inner=M2_inner, R_outer=M2_outer
        )(x, y)

    M2_z_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M2_dzdx_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M2_dzdy_grid = np.empty((nmode, args.ngrid, args.ngrid))
    M2_d2zdxy_grid = np.empty((nmode, args.ngrid, args.ngrid))

    z2 = Udn3norm.T - zk_eval2

    m2gridder = Gridder(x, y, args.ngrid, M2_outer, interp='z')
    for imode in tqdm(range(nmode)):
        m2_result = m2gridder.grid(z2[imode])
        M2_z_grid[imode] = m2_result[0]
        M2_dzdx_grid[imode] = m2_result[1]
        M2_dzdy_grid[imode] = m2_result[2]
        M2_d2zdxy_grid[imode] = m2_result[3]

    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        def colorbar(mappable):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.pyplot as plt
            last_axes = plt.gca()
            ax = mappable.axes
            fig = ax.figure
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(mappable, cax=cax)
            plt.sca(last_axes)
            return cbar
        fig, axes = plt.subplots(
            nrows=4, ncols=5, figsize=(11, 7.5),
            sharex=True, sharey=True
        )
        vmax = 1e0
        xs = np.linspace(-2.0, 2.0, 1000)
        xs, ys = np.meshgrid(xs, xs)
        rs = np.hypot(xs, ys)
        w = rs <= 1.71
        w &= w > 0.9
        for imode, ax in enumerate(axes.ravel()):
            grid = batoid.Bicubic(
                m2gridder.x_grid, m2gridder.x_grid,
                M2_z_grid[imode], M2_dzdx_grid[imode], M2_dzdy_grid[imode], M2_d2zdxy_grid[imode],
                nanpolicy='zero'
            )
            out = np.full_like(xs, np.nan)
            out[w] = grid.sag(xs[w], ys[w])
            cbar = colorbar(ax.imshow(
                out*1e6, vmin=-vmax, vmax=vmax,
                cmap='seismic', origin='lower',
                extent=[-2, 2, -2, 2],
            ))
            ax.scatter(
                x, y, c=z2[imode]*1e6,
                edgecolor='k', lw=0.1,
                vmin=-vmax, vmax=vmax, cmap='seismic', s=10
            )

            cbar.set_label("microns", fontsize=6)
            cbar.ax.tick_params(labelsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.add_patch(
                Circle((0, 0), 1.71, facecolor='None', edgecolor='black', lw=0.1)
            )
            ax.add_patch(
                Circle((0, 0), 0.9, facecolor='None', edgecolor='black', lw=0.1)
            )
            ax.set_title(f"Mode {imode:d}", fontsize=8)

        fig.tight_layout()
        plt.show()

    with open(args.output, 'wb') as f:
        pickle.dump(
            (M2zk,
             m2gridder.x_grid, M2_z_grid, M2_dzdx_grid, M2_dzdy_grid, M2_d2zdxy_grid),
            f
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M2_sag.pkl",
        help="Input pickle or mat file.  Default: M2_sag.pkl"
    )
    parser.add_argument(
        "output",
        default="M2_decomposition.pkl",
        help=
            "output file name  "
            "Default: M2_decomposition.pkl",
        nargs='?'
    )
    parser.add_argument(
        "--circular",
        action="store_true",
        help=
            "Use circular (instead of annular) Zernike decomposition.  "
            "Default: False"
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results.  Default: False"
    )
    args = parser.parse_args()
    main(args)
