import numpy as np
import asdf
from scipy.io import loadmat
import matplotlib.pyplot as plt


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


def main(args):
    with asdf.open(args.input) as af:
        ax = np.array(af['actuators']['X_Position'])
        ay = np.array(af['actuators']['Y_Position'])
        x = np.array(af['fea_nodes']['X_Position'])
        y = np.array(af['fea_nodes']['Y_Position'])
        norm = np.array(af['bend_1um']['normal'])[args.imode]*1e6 # m -> um
        force = np.array(af['bend_1um']['force'])[args.imode]
    m1m3_mat_data = loadmat(args.compare)

    udata = m1m3_mat_data['Udn3norm'][:, args.imode]*1e6 # m -> um
    vdata = m1m3_mat_data['Vdn3norm'][:, args.imode] # Newtons

    if np.dot(udata, norm) < 0:
        udata *= -1
        vdata *= -1
    vmax = np.max(np.abs(np.quantile(udata, [0.01, 0.99])))
    diff = udata - norm
    vdmax = np.max(np.abs(np.quantile(diff, [0.01, 0.99])))

    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(10, 7),
        sharex=True, sharey=True
    )
    colorbar(axes[0,0].scatter(
        m1m3_mat_data['x'][:, 0], m1m3_mat_data['y'][:, 0],
        c=udata,
        cmap='seismic',
        vmin=-vmax, vmax=vmax, s=1
    )).set_label("um")
    colorbar(axes[0,1].scatter(
        x, y,
        c=norm,
        cmap='seismic',
        vmin=-vmax, vmax=vmax, s=1
    )).set_label("um")
    colorbar(axes[0,2].scatter(
        x, y,
        c=diff,
        cmap='seismic',
        vmin=-vdmax, vmax=vdmax, s=1
    )).set_label("um")
    axes[0,0].set_title("bxin")
    axes[0,1].set_title("jmeyers")
    axes[0,2].set_title("diff")
    axes[1,0].set_title("bxin")
    axes[1,1].set_title("jmeyers")
    axes[1,2].set_title("diff")

    fmax = np.max(np.abs(np.quantile(vdata, [0.01, 0.99])))
    fdmax = np.max(np.abs(np.quantile(vdata-force, [0.01, 0.99])))
    colorbar(axes[1,0].scatter(
        ax, ay,
        c=force,
        cmap='seismic',
        vmin=-fmax, vmax=fmax, s=10
    )).set_label("Newtons")
    colorbar(axes[1,1].scatter(
        ax, ay,
        c=vdata,
        cmap='seismic',
        vmin=-fmax, vmax=fmax, s=10
    )).set_label("Newtons")
    colorbar(axes[1,2].scatter(
        ax, ay,
        c=vdata-force,
        cmap='seismic',
        vmin=-fdmax, vmax=fdmax, s=10
    )).set_label("Newtons")

    fig.suptitle(f"mode {args.imode}")
    for ax in axes.ravel():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="M1M3_NASTRAN.asdf",
        help="Input asdf file.  Default: M1M3_NASTRAN.asdf"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default= "/Users/josh/src/M1M3_ML/data/myUdn3norm_156.mat",
        help="Comparison matlab file."
    )
    parser.add_argument("--imode", type=int, default=0, help="Mode number to plot.")
    args = parser.parse_args()
    main(args)
