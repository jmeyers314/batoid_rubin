import batoid
import galsim
import matplotlib.pyplot as plt
import numpy as np

from tools import getFocusedTelescope, spotSize


class Callback:
    def __init__(self, title):
        self.title = title

    def __call__(self, fig, axes):
        fig.suptitle(
            self.title,
            fontdict={'fontname': 'monospace', 'weight': 'bold'},
            fontsize=12,
            weight='bold'
        )


for f in "ugrizy":
    bandpass = galsim.Bandpass(f"LSST_{f}.dat", wave_type='nm')
    wavelength = bandpass.effective_wavelength * 1e-9

    # telescope = batoid.Optic.fromYaml(f"Rubin_v3.12_{f}.yaml")
    telescope = batoid.Optic.fromYaml(f"LSST_{f}.yaml")
    telescope = getFocusedTelescope(telescope, wavelength)
    spot_size = spotSize(telescope, wavelength)

    thx, thy = batoid.utils.hexapolar(
        outer=np.deg2rad(1.75), inner=0.0,
        nrad=7, naz=45,
    )

    zk = np.zeros((len(thx), 29))
    for i, (thx_, thy_) in enumerate(zip(thx, thy)):
        zk[i] = batoid.zernike(
            telescope, thx_, thy_, wavelength, jmax=28, eps=0.612, nx=128
        ) * wavelength * 1e6  # convert to microns

    # Compute double Zernike coefficients
    bases = galsim.zernike.zernikeBasis(36, thx, thy, R_outer=np.deg2rad(1.75))
    dzs, *_ = np.linalg.lstsq(bases.T, zk, rcond=None)
    dzs[:, :4] = 0.0  # Zero out PTT
    dzs[0, :] = 0.0  # k=0 is unused
    # What are the dominant remaining terms?
    asort = np.argsort(np.square(dzs).ravel())[::-1]
    ks, js = np.unravel_index(asort[:20], dzs.shape)
    cumsum = 0.0
    for k, j in zip(ks, js):
        val = dzs[k, j]
        cumsum += val**2
        print("{:3d} {:3d} {:8.4f} {:8.4f}".format(k, j, val, np.sqrt(cumsum)))
    print("sum sqr dz {:8.4f}".format(np.sqrt(np.sum(dzs**2))))

    fig = plt.figure(figsize=(13, 8))
    title = f"LSST {f} band, {int(wavelength*1e9):d} nm"
    batoid.plotUtils.zernikePyramid(
        thx, thy, zk[:, 4:].T, fig=fig, vmin=-1, vmax=1, s=3,
        callback=Callback(title),
    )
    cumsum = 0.0
    for i, (k, j) in enumerate(zip(ks, js)):
        val = dzs[k, j]
        cumsum += val**2
        fig.text(
            0.82, 0.9-0.015*i,
            f"{k:>3d} {j:>3d} {val:8.4f} {np.sqrt(cumsum):8.4f}",
            fontdict={'fontname': 'Courier New', 'weight': 'bold'}
        )
    fig.text(
        0.82, 0.9+0.03,
        "  k   j      val   cumsum",
        fontdict={'fontname': 'Courier New', 'weight': 'bold'}
    )
    fig.text(
        0.82, 0.9+0.015,
        "-------------------------",
        fontdict={'fontname': 'Courier New', 'weight': 'bold'}
    )
    fig.text(
        0.03, 0.93,
        f"RMS spot size: {spot_size:.3f} arcsec",
        fontdict={'fontname': 'Courier New', 'weight': 'bold'}
    )

    plt.savefig(f"intrinsic_{f}.png")
