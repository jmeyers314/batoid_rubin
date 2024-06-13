# TODO:
#       normalize by RMS T or RMS sqrt(T)
#       force output in builder.py
#       flesh out comcam

# Use fiducial modes instead of raw.
# Manually querying forces so need mapping below.
M1M3_MODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26]  # Swap 26 -> 19
M2_MODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27]  # Swap 25,26,27 -> 17,18,19

from pathlib import Path

import batoid
import galsim
# Set the tick labels to bold
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from batoid_rubin import datadir
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

from tools import getFocusedTelescope, withPerturbationAmplitude, spotSize

mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['font.family'] = 'monospace'
mpl.rcParams['font.weight'] = 'bold'


def colorbar(mappable):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(
        mappable,
        cax=cax,
    )
    cbar.ax.tick_params(
        labelsize=7,
        labelfontfamily='monospace',
    )
    plt.sca(last_axes)
    return cbar


class Callback:
    def __init__(self, title):
        self.title = title

    def __call__(self, fig, axes):
        for j, ax in axes.items():
            if j in [2, 3]:
                ax.set_title(
                    f"Z{j} (arcsec)",
                    fontdict={'fontname': 'monospace', 'weight': 'bold'},
                    fontsize=10
                )
                scatter = ax.get_children()[0]
                scatter.set_clim(-1, 1)
                # Convert microns -> arcsec for Rubin tip/tilt
                scatter.get_array()[:] *= 0.08417817847461787
            else:
                ax.set_title(
                    f"Z{j} (µm)",
                    fontdict={'fontname': 'monospace', 'weight': 'bold'},
                    fontsize=10
                )
        fig.suptitle(
            self.title,
            fontdict={'fontname': 'monospace', 'weight': 'bold'},
            fontsize=12,
            weight='bold'
        )


def rms(arr):
    return np.sqrt(np.nanmean(arr**2))


def main(args):
    dof_names = [
        'M2 dz',
        'M2 dx',
        'M2 dy',
        'M2 Rx',
        'M2 Ry',
        'Cam dz',
        'Cam dx',
        'Cam dy',
        'Cam Rx',
        'Cam Ry',
    ]
    dof_names += [f'M1M3 mode {i}' for i in range(20)]
    dof_names += [f'M2 mode {i}' for i in range(20)]

    # Hard code for now...
    m1m3_forces = fits.getdata(Path(datadir)/"bend"/"M1M3_bend_forces.fits.gz")
    m2_forces = fits.getdata(Path(datadir)/"bend"/"M2_bend_forces.fits.gz")
    m1m3_actuators = fits.getdata(Path(datadir)/"bend"/"M1M3_actuator_table.fits.gz")
    m2_actuators = fits.getdata(Path(datadir)/"bend"/"M2_actuator_table.fits.gz")

    bandpass = galsim.Bandpass(f"LSST_r.dat", wave_type='nm')
    wavelength = bandpass.effective_wavelength * 1e-9

    # Load and focus the telescope
    # telescope = batoid.Optic.fromYaml(f"Rubin_v3.12_r.yaml")
    telescope = batoid.Optic.fromYaml(f"LSST_r.yaml")
    telescope = getFocusedTelescope(telescope, wavelength)
    spot_size = spotSize(telescope, wavelength)
    print(f"Found focused spot size: {spot_size:.3f} arcsec")

    sag_thx = np.linspace(-4.18, 4.18, 128)
    sag_thx, sag_thy = np.meshgrid(sag_thx, sag_thx)
    sag_r2 = sag_thx**2 + sag_thy**2
    w1 = (2.558**2 <= sag_r2) & (sag_r2 <= 4.18**2)
    w3 = (0.55**2 <= sag_r2) & (sag_r2 <= 2.508**2)
    sag13 = telescope['M1'].surface.sag(sag_thx, sag_thy)
    sag13[~w1] = np.nan
    sag3 = telescope['M3'].surface.sag(sag_thx, sag_thy)
    sag13[w3] = sag3[w3]
    sag13 *= 1e6  # convert to microns

    sag2_thx = np.linspace(-1.7, 1.7, 128)
    sag2_thx, sag2_thy = np.meshgrid(sag2_thx, sag2_thx)
    sag2_r2 = sag2_thx**2 + sag2_thy**2
    w2 = (0.9**2 <= sag2_r2) & (sag2_r2 <= 1.71**2)
    sag2 = np.zeros_like(sag13)
    sag2[w2] = telescope['M2'].surface.sag(sag2_thx[w2], sag2_thy[w2])
    sag2 *= 1e6  # convert to microns

    if args.fast:
        nrad, naz, nx = 5, 30, 32
    else:
        nrad, naz, nx = 15, 90, 128

    thx, thy = batoid.utils.hexapolar(
        outer=np.deg2rad(1.75), inner=0.0,
        nrad=nrad, naz=naz,
    )

    # Get intrinsic Zernike coefficients
    print("Collecting intrinsic LSSTCam and ComCam Zernike coefficients")
    zk0 = np.zeros((len(thx), 29))
    for i, (thx_, thy_) in enumerate(zip(tqdm(thx), thy)):
        zk0[i] = batoid.zernike(
            telescope, thx_, thy_, wavelength, jmax=28, eps=0.612, nx=nx
        ) * wavelength * 1e6  # convert to microns

    # Get ComCam points
    thxmax = np.deg2rad(1.5*0.2*4000/3600)
    thx_comcam = np.linspace(-thxmax, thxmax, nrad)
    thx_comcam, thy_comcam = np.meshgrid(thx_comcam, thx_comcam)

    # Get intrinsic Zernike coefficients for ComCam
    zk0_comcam = np.zeros((len(thx_comcam.ravel()), 29))
    for i, (thx_, thy_) in enumerate(zip(
        tqdm(thx_comcam.ravel()),
        thy_comcam.ravel()
    )):
        zk0_comcam[i] = batoid.zernike(
            telescope, thx_, thy_, wavelength, jmax=28, eps=0.612, nx=nx
        ) * wavelength * 1e6  # convert to microns

    # Iterate degrees of freedom
    idofs = [args.dof] if args.dof is not None else range(50)
    for idof in idofs:
        print(f"Working on degree of freedom {idof}")
        # iterate amplitude until 80-centile spot size is 0.1 arcsec
        perturbed, amplitude = withPerturbationAmplitude(
            telescope, idof, 0.1, wavelength
        )
        new_spot_size = spotSize(perturbed, wavelength)
        print(f"Found perturbation amplitude: {amplitude:.3f}")
        print(f"New spot size: {new_spot_size:.3f} arcsec")

        # Get perturbed Zernike coefficients
        zk = np.zeros((len(thx), 29))
        for i, (thx_, thy_) in enumerate(zip(tqdm(thx), thy)):
            zk[i] = batoid.zernike(
                perturbed, thx_, thy_, wavelength, jmax=28, eps=0.612, nx=nx
            ) * wavelength * 1e6  # convert to microns
        delta_zk = (zk - zk0) # Zk change corresponding to 0.1 arcsec spot size

        # Compute double Zernike coefficients
        bases = galsim.zernike.zernikeBasis(28, thx, thy, R_outer=np.deg2rad(1.75))
        dzs, *_ = np.linalg.lstsq(bases.T, delta_zk, rcond=None)
        dzs[:, :4] = 0.0  # Zero out PTT
        dzs[0, :] = 0.0  # k=0 is unused

        rms_lsstcam = np.sqrt(np.sum(dzs[:, 4:]**2))
        print(f"RMS WFE: {rms_lsstcam:.4f}")

        fig = plt.figure(figsize=(13, 8))
        title = dof_names[idof]
        batoid.plotUtils.zernikePyramid(
            thx, thy, delta_zk[:, 2:].T, jmin=2,
            fig=fig, s=2,
            vmin=-1, vmax=1, vdim=False,
            callback=Callback(title)
        )
        # What are the dominant remaining terms?
        asort = np.argsort(np.square(dzs).ravel())[::-1]
        ks, js = np.unravel_index(asort[:20], dzs.shape)
        cumsum = 0.0
        for i, (k, j) in enumerate(zip(ks, js)):
            val = dzs[k, j]
            cumsum += val**2
            fig.text(
                0.82, 0.9-0.015*i,
                f"{k:>3d} {j:>3d} {val:8.4f} {np.sqrt(cumsum):8.4f}",
            )
        fig.text(
            0.82, 0.9+0.03,
            "  k   j      val   cumsum",
        )
        fig.text(
            0.82, 0.9+0.015,
            "-------------------------",
        )
        fig.text(
            0.04, 0.9+0.03,
            f"Spot Size: {new_spot_size:.4f} arcsec"
        )
        fig.text(
            0.04, 0.9+0.015,
            f"RMS: {rms_lsstcam:.4f} µm",
        )
        unit = "µm"
        if idof in [3,4,8,9]:
            unit = "arcsec"
        fig.text(
            0.04, 0.9,
            f"perturbation: {amplitude:.4f} {unit}",
        )
        if 9 < idof <= 29:
            fig.text(
                0.04, 0.9-0.015,
                f"RMS force: {rms(m1m3_forces[M1M3_MODES[idof-10]])*amplitude:.4f} N",
            )
            fig.text(
                0.04, 0.9-0.03,
                f"PTP force: {np.ptp(m1m3_forces[M1M3_MODES[idof-10]])*amplitude:.4f} N",
            )
            ax = fig.add_axes([0.025, 0.4, 0.12, 0.12])
            colorbar(ax.scatter(
                m1m3_actuators['X_Position'],
                m1m3_actuators['Y_Position'],
                c=m1m3_forces[M1M3_MODES[idof-10]]*amplitude,
                s=10, cmap='seismic',
                norm=SymLogNorm(linthresh=10, vmin=-10000.0, vmax=10000.0)
            ))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M1M3 forces (N)", fontsize=7, fontweight='bold')
        elif 29 < idof <= 49:
            fig.text(
                0.04, 0.9-0.015,
                f"RMS force: {rms(m2_forces[M2_MODES[idof-30]])*amplitude:.4f} N",
            )
            fig.text(
                0.04, 0.9-0.03,
                f"PTP force: {np.ptp(m2_forces[M2_MODES[idof-30]])*amplitude:.4f} N",
            )
            ax = fig.add_axes([0.025, 0.4, 0.12, 0.12])
            colorbar(ax.scatter(
                m2_actuators['X_Position'],
                m2_actuators['Y_Position'],
                c=m2_forces[M2_MODES[idof-30]]*amplitude,
                s=10, cmap='seismic',
                norm=SymLogNorm(linthresh=1, vmin=-1000.0, vmax=1000.0)
            ))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M2 forces (N)", fontsize=7, fontweight='bold')

        # Mirror figure
        if 10 <= idof <= 29:
            ax = fig.add_axes([0.85, 0.4, 0.12, 0.12])
            sag = perturbed['M1'].surface.sag(sag_thx, sag_thy)
            sag[~w1] = np.nan
            sag3 = perturbed['M3'].surface.sag(sag_thx, sag_thy)
            sag[w3] = sag3[w3]
            sag *= 1e6  # convert to microns
            sag -= sag13
            colorbar(ax.imshow(sag, vmin=-2, vmax=2, cmap='seismic'))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M1M3 surface (µm)", fontsize=7, fontweight='bold')
        if 30 <= idof <= 49:
            ax = fig.add_axes([0.85, 0.4, 0.12, 0.12])
            sag = perturbed['M2'].surface.sag(sag2_thx, sag2_thy)
            sag[~w2] = np.nan
            sag *= 1e6
            sag -= sag2
            colorbar(ax.imshow(sag, vmin=-2, vmax=2, cmap='seismic'))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M2 surface (µm)", fontsize=7, fontweight='bold')

        # Spot diagrams
        ax = fig.add_axes([0.66, 0.75, 0.18, 0.18])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        spotx = []
        spoty = []
        for thr in [0.0, 0.875, 1.75]:
            if thr == 0.0:
                thphs = [0.0]
            elif thr == 0.875:
                thphs = np.linspace(0, 2*np.pi, 6, endpoint=False)
            elif thr == 1.75:
                thphs = np.linspace(0, 2*np.pi, 12, endpoint=False)
            thx_ = thr*np.cos(thphs)
            thy_ = thr*np.sin(thphs)
            for thx__, thy__ in zip(thx_, thy_):
                spot = batoid.analysis.spot(
                    perturbed,
                    np.deg2rad(thx__), np.deg2rad(thy__),
                    wavelength=wavelength,
                )
                scale = 20e-6
                spotx.append(spot[0]/scale+3*thx__/1.75)
                spoty.append(spot[1]/scale+3*thy__/1.75)
        spotx = np.concatenate(spotx)
        spoty = np.concatenate(spoty)
        ax.scatter(spotx, spoty, s=0.1, alpha=0.1, c='k')

        # Label Camera
        fig.text(0.04, 0.96, "LSSTCam", fontsize=12)

        plt.savefig(f"LSSTCam_dof_{idof:02d}.png", dpi=300)
        plt.close(fig)


        # ComCam
        comcam_spot_size = spotSize(perturbed, wavelength, outer_field=1.75/5)
        # Now get ComCam zernikes
        zk_comcam = np.zeros((len(thx_comcam.ravel()), 29))
        for i, (thx_, thy_) in enumerate(zip(
            tqdm(thx_comcam.ravel()), thy_comcam.ravel()
        )):
            zk_comcam[i] = batoid.zernike(
                perturbed, thx_, thy_, wavelength, jmax=28, eps=0.612, nx=nx
            ) * wavelength * 1e6  # convert to microns
        delta_zk_comcam = (zk_comcam - zk0_comcam)

        rms_comcam = np.sqrt(np.sum(np.mean(delta_zk_comcam[:, 4:]**2, axis=0)))

        # Fit ComCam double Zernikes using LSSTCam basis
        # A little suspect since ComCam FOV is a square
        bases = galsim.zernike.zernikeBasis(
            28, thx_comcam.ravel(), thy_comcam.ravel(), R_outer=thxmax
        )
        dzs_comcam, *_ = np.linalg.lstsq(bases.T, delta_zk_comcam, rcond=None)
        dzs_comcam[:, :4] = 0.0  # Zero out PTT
        dzs_comcam[0, :] = 0.0  # k=0 is unused

        # Now repeat the plot, but for ComCam field of view.
        title = dof_names[idof]
        fig = plt.figure(figsize=(13, 8))
        batoid.plotUtils.zernikePyramid(
            thx_comcam, thy_comcam, delta_zk_comcam[:, 2:].T, jmin=2,
            fig=fig, s=3,
            vmin=-1, vmax=1, vdim=False,
            callback=Callback(title)
        )
        # What are the dominant remaining terms?
        asort = np.argsort(np.square(dzs_comcam).ravel())[::-1]
        ks, js = np.unravel_index(asort[:20], dzs_comcam.shape)
        cumsum = 0.0
        for i, (k, j) in enumerate(zip(ks, js)):
            val = dzs_comcam[k, j]
            cumsum += val**2
            fig.text(
                0.82, 0.9-0.015*i,
                f"{k:>3d} {j:>3d} {val:8.4f} {np.sqrt(cumsum):8.4f}",
            )
        fig.text(
            0.82, 0.9+0.03,
            "  k   j      val   cumsum",
        )
        fig.text(
            0.82, 0.9+0.015,
            "-------------------------",
        )
        fig.text(
            0.04, 0.9+0.03,
            f"Spot Size: {comcam_spot_size:.4f} arcsec"
        )
        fig.text(
            0.04, 0.9+0.015,
            f"RMS: {rms_comcam:.4f} µm",
        )
        fig.text(
            0.04, 0.9,
            f"perturbation: {amplitude:.4f} {unit}",
        )
        if 9 < idof <= 29:
            fig.text(
                0.04, 0.9-0.015,
                f"RMS force: {rms(m1m3_forces[M1M3_MODES[idof-10]])*amplitude:.4f} N",
            )
            fig.text(
                0.04, 0.9-0.03,
                f"PTP force: {np.ptp(m1m3_forces[M1M3_MODES[idof-10]])*amplitude:.4f} N",
            )
            ax = fig.add_axes([0.025, 0.4, 0.12, 0.12])
            colorbar(ax.scatter(
                m1m3_actuators['X_Position'],
                m1m3_actuators['Y_Position'],
                c=m1m3_forces[M1M3_MODES[idof-10]]*amplitude,
                s=10, cmap='seismic',
                norm=SymLogNorm(linthresh=10, vmin=-10000.0, vmax=10000.0)
            ))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M1M3 forces (N)", fontsize=7, fontweight='bold')

        elif 29 < idof <= 49:
            fig.text(
                0.04, 0.9-0.015,
                f"RMS force: {rms(m2_forces[M2_MODES[idof-30]])*amplitude:.4f} N",
            )
            fig.text(
                0.04, 0.9-0.03,
                f"PTP force: {np.ptp(m2_forces[M2_MODES[idof-30]])*amplitude:.4f} N",
            )
            ax = fig.add_axes([0.025, 0.4, 0.12, 0.12])
            colorbar(ax.scatter(
                m2_actuators['X_Position'],
                m2_actuators['Y_Position'],
                c=m2_forces[M2_MODES[idof-30]]*amplitude,
                s=10, cmap='seismic',
                norm=SymLogNorm(linthresh=1, vmin=-1000.0, vmax=1000.0)
            ))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M2 forces (N)", fontsize=7, fontweight='bold')

        # Mirror figure
        if 10 <= idof <= 29:
            ax = fig.add_axes([0.85, 0.4, 0.12, 0.12])
            sag = perturbed['M1'].surface.sag(sag_thx, sag_thy)
            sag[~w1] = np.nan
            sag3 = perturbed['M3'].surface.sag(sag_thx, sag_thy)
            sag[w3] = sag3[w3]
            sag *= 1e6  # convert to microns
            sag -= sag13
            colorbar(ax.imshow(sag, vmin=-2, vmax=2, cmap='seismic'))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M1M3 surface (µm)", fontsize=7, fontweight='bold')
        if 30 <= idof <= 49:
            ax = fig.add_axes([0.85, 0.4, 0.12, 0.12])
            sag = perturbed['M2'].surface.sag(sag2_thx, sag2_thy)
            sag[~w2] = np.nan
            sag *= 1e6
            sag -= sag2
            colorbar(ax.imshow(sag, vmin=-2, vmax=2, cmap='seismic'))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("M2 surface (µm)", fontsize=7, fontweight='bold')

        # Spot diagrams
        ax = fig.add_axes([0.66, 0.75, 0.18, 0.18])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        spotx = []
        spoty = []
        thx_ = np.linspace(-1, 1, 3)*4000*0.2/3600
        thx_, thy_ = np.meshgrid(thx_, thx_)
        for thx__, thy__ in zip(thx_.ravel(), thy_.ravel()):
            spot = batoid.analysis.spot(
                perturbed,
                np.deg2rad(thx__), np.deg2rad(thy__),
                wavelength=wavelength,
            )
            scale = 20e-6
            spotx.append(spot[0]/scale+6*thx__)
            spoty.append(spot[1]/scale+6*thy__)
        spotx = np.concatenate(spotx)
        spoty = np.concatenate(spoty)
        ax.scatter(spotx, spoty, s=0.1, alpha=0.1, c='k')

        fig.text(0.04, 0.96, "ComCam", fontsize=12)

        plt.savefig(f"ComCam_dof_{idof:02d}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dof", type=int, default=None)
    parser.add_argument("--fast", action='store_true')
    args = parser.parse_args()
    main(args)
