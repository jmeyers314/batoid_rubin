from batoid_rubin import LSSTBuilder
import batoid
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar


def spotSize(optic, wavelength, nrad=5, naz=30, outer_field=1.75):
    """Compute the RMS spot size for a given optic.

    Parameters
    ----------
    optic : batoid.Optic
        Optic for which to compute the RMS spot size.
    wavelength : float
        Wavelength in meters.
    nrad : int, optional
        Number of pupil/field points in radial direction.
    naz : int, optional
        Number of pupil/field points in azimuthal direction.
    outer_field : float, optional
        Outer radius of field in degrees.

    Returns
    -------
    rms : float
        RMS spot size in arcseconds.
    """
    # Get the field points
    thx, thy = batoid.utils.hexapolar(
        outer=np.deg2rad(outer_field),
        inner=0.0,
        nrad=nrad,
        naz=naz,
    )

    # We'll use the 80th percentile of the RMS spot size over the field points
    mean_square = []
    for thx_, thy_ in zip(thx, thy):
        rays = batoid.RayVector.asPolar(
            optic,
            wavelength=wavelength,
            theta_x=thx_,
            theta_y=thy_,
            nrad=nrad * 3,
            naz=naz * 3,
        )
        rays = optic.trace(rays)
        xs = rays.x[~rays.vignetted]
        ys = rays.y[~rays.vignetted]
        xs -= np.mean(xs)
        ys -= np.mean(ys)
        mean_square.append(np.mean(np.square(xs)) + np.mean(np.square(ys)))
    return np.sqrt(np.quantile(mean_square, 0.8)) * 0.2 / 10e-6  # convert to arcsec


def spotSizeObjective(camera_z, optic, wavelength, nrad=5, naz=30, outer_field=1.75):
    perturbed = optic.withGloballyShiftedOptic("LSSTCamera", [0, 0, camera_z])
    return spotSize(perturbed, wavelength, nrad, naz, outer_field)


def getFocusedTelescope(telescope, wavelength, nrad=5, naz=30, outer_field=1.75):
    result = minimize_scalar(
        spotSizeObjective,
        bounds=(-1e-4, 1e-4),
        args=(telescope, wavelength, nrad, naz, outer_field),
        options={"xatol": 1e-8},
    )
    focus_z = result.x
    return telescope.withGloballyShiftedOptic("LSSTCamera", [0, 0, focus_z])


def amplitudeObjective(
    amplitude, idof, target, optic, wavelength, nrad=5, naz=30, outer_field=1.75,
    builder_kwargs=None
):
    if builder_kwargs is None:
        builder_kwargs = {}
    builder = LSSTBuilder(optic, **builder_kwargs)
    dof = np.zeros_like(builder.dof)
    dof[idof] = amplitude
    perturbed = builder.with_aos_dof(dof).build()
    return spotSize(perturbed, wavelength, nrad, naz, outer_field) - target


def bracket(f, a, b, args=(), fixa=False):
    if b < a:
        a, b = b, a
    fa = f(a, *args)
    fb = f(b, *args)
    while fa*fb > 0:
        if abs(fa) < abs(fb) and not fixa:
            a -= b-a
            fa = f(a, *args)
        else:
            b += b-a
            fb = f(b, *args)

    return a, b


def withPerturbationAmplitude(
    telescope, idof, target, wavelength, nrad=5, naz=30,
    outer_field=1.75, builder_kwargs=None
):
    if builder_kwargs is None:
        builder_kwargs = {}
    args = (idof, target, telescope, wavelength, nrad, naz, outer_field, builder_kwargs)
    a, b = bracket(
        amplitudeObjective,
        0.0, 0.1,
        args=args,
        fixa=True
    )
    amplitude_result = root_scalar(
        amplitudeObjective,
        args=args,
        bracket=(a, b),
        method='brentq',
        xtol=1e-8
    )
    amplitude = amplitude_result.root
    builder = LSSTBuilder(telescope, **builder_kwargs)
    dof = np.zeros_like(builder.dof)
    dof[idof] = amplitude
    perturbed = builder.with_aos_dof(dof).build()
    return perturbed, amplitude
