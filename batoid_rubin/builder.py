from pathlib import Path
from collections import namedtuple
from copy import copy
from functools import lru_cache
from numbers import Real

import astropy.io.fits as fits
import batoid
import galsim
import numpy as np
import yaml

from .utils import _node_to_grid, _fits_cache, attach_attr


BendingMode = namedtuple(
    "BendingMode",
    "zk R_outer R_inner x y z dzdx dzdy d2zdxy"
)


RealizedBend = namedtuple(
    "RealizedBend",
    "zk grid"
)


@lru_cache(maxsize=2)
def m1m3_fea_nodes(fea_dir):
    data = fits.getdata(Path(fea_dir) / "M1M3_1um_156_grid.fits.gz")
    idx = data[:, 0]
    bx = data[:, 1]  # (5256,)
    by = data[:, 2]
    idx1 = (idx == 1)
    idx3 = (idx == 3)
    bx.flags.writeable = False
    by.flags.writeable = False
    idx1.flags.writeable = False
    idx3.flags.writeable = False
    return bx, by, idx1, idx3


@lru_cache(maxsize=4)
def m1m3_grid_xy(bend_dir):
    with open(Path(bend_dir) / "bend.yaml") as f:
        config = yaml.safe_load(f)
    m1_grid_xy = fits.getdata(Path(bend_dir) / config['M1']['grid']['coords'])
    m3_grid_xy = fits.getdata(Path(bend_dir) / config['M3']['grid']['coords'])
    m1_grid_xy.flags.writeable = False
    m3_grid_xy.flags.writeable = False
    return m1_grid_xy, m3_grid_xy


@lru_cache(maxsize=2)
def m2_fea_nodes(fea_dir):
    data = fits.getdata(Path(fea_dir) / "M2_1um_grid.fits.gz")
    bx = data[:, 1]  # meters
    by = data[:, 2]
    bx.flags.writeable = False
    by.flags.writeable = False
    return bx, by


@lru_cache(maxsize=4)
def m2_grid_xy(bend_dir):
    with open(Path(bend_dir) / "bend.yaml") as f:
        config = yaml.safe_load(f)
    m2_grid_xy = fits.getdata(Path(bend_dir) / config['M2']['grid']['coords'])
    m2_grid_xy.flags.writeable = False
    return m2_grid_xy


@lru_cache(maxsize=16)
def m1m3_gravity(fea_dir, optic, zenith):
    zdata = _fits_cache(fea_dir, "M1M3_dxdydz_zenith.fits.gz")
    hdata = _fits_cache(fea_dir, "M1M3_dxdydz_horizon.fits.gz")

    if zenith is None:
        return np.zeros_like(zdata[:,2])

    dxyz = (
        zdata * np.cos(zenith) +
        hdata * np.sin(zenith)
    )
    dz = dxyz[:,2]

    # Interpolate these node displacements into z-displacements at
    # original node x/y positions.
    bx, by, idx1, idx3 = m1m3_fea_nodes(fea_dir)

    # M1
    zRef = optic['M1'].surface.sag(bx[idx1], by[idx1])
    zpRef = optic['M1'].surface.sag(
        (bx+dxyz[:, 0])[idx1],
        (by+dxyz[:, 1])[idx1]
    )
    dz[idx1] += zRef - zpRef

    # M3
    zRef = optic['M3'].surface.sag(bx[idx3], by[idx3])
    zpRef = optic['M3'].surface.sag(
        (bx+dxyz[:, 0])[idx3],
        (by+dxyz[:, 1])[idx3]
    )
    dz[idx3] += zRef - zpRef

    # Subtract PTT
    # This kinda makes sense for M1, but why for combined M1M3?
    zBasis = galsim.zernike.zernikeBasis(
        3, bx, by, R_outer=4.18, R_inner=2.558
    )
    coefs, *_ = np.linalg.lstsq(zBasis.T, dxyz[:, 2], rcond=None)
    zern = galsim.zernike.Zernike(coefs, R_outer=4.18, R_inner=2.558)
    dz -= zern(bx, by)

    dz.flags.writeable = False
    return dz


@lru_cache(maxsize=16)
def m1m3_temperature(fea_dir, TBulk, TxGrad, TyGrad, TzGrad, TrGrad):
    tbdz, txdz, tydz, tzdz, trdz = _fits_cache(
        fea_dir,
        "M1M3_thermal_FEA.fits.gz"
    )

    out = TBulk * tbdz
    out += TxGrad * txdz
    out += TyGrad * tydz
    out += TzGrad * tzdz
    out += TrGrad * trdz
    out *= 1e-6
    out.flags.writeable = False
    return out


@lru_cache(maxsize=16)
def m1m3_lut(fea_dir, zenith, error, seed):
    if zenith is None:
        return np.zeros(256)

    from scipy.interpolate import interp1d
    data = _fits_cache(fea_dir, "M1M3_LUT.fits.gz")

    LUT_force = interp1d(data[0], data[1:])(zenith.deg)

    if error != 0.0:
        # Get current forces so we can rebalance after applying random error
        z_force = np.sum(LUT_force[:156])
        y_force = np.sum(LUT_force[156:])

        rng = np.random.default_rng(seed)
        LUT_force *= rng.uniform(1-error, 1+error, size=len(LUT_force))

        # Balance forces by adjusting means in 2 ranges
        LUT_force[:156] -= np.mean(LUT_force[:156]) - z_force
        LUT_force[156:] -= np.mean(LUT_force[156:]) - y_force

        # # Balance forces by manipulating these 2 actuators
        # LUT_force[155] = z_force - np.sum(LUT_force[:155])
        # LUT_force[-1] = y_force - np.sum(LUT_force[156:-1])

    zf = _fits_cache(fea_dir, "M1M3_force_zenith.fits.gz")
    hf = _fits_cache(fea_dir, "M1M3_force_horizon.fits.gz")
    u0 = zf * np.cos(zenith)
    u0 += hf * np.sin(zenith)

    out = LUT_force - u0
    out.flags.writeable = False
    return out


def m1m3_force_to_surface(fea_dir, forces):
    G = _fits_cache(fea_dir, "M1M3_influence_256.fits.gz")
    out = np.dot(G, forces)
    out.flags.writeable = False
    return out


def transform_zernike(zernike, R_outer, R_inner):
    xy = zernike._coef_array_xy
    ret = galsim.zernike.Zernike.__new__(galsim.zernike.Zernike)
    ret._coef_array_xy = xy
    ret.R_outer = R_outer
    ret.R_inner = R_inner
    return ret


def _load_mirror_bend(bend_dir, inds, config):
    zk = fits.getdata(Path(bend_dir) / config['zk']['file'])
    grid = fits.getdata(Path(bend_dir) / config['grid']['file'])
    coords = fits.getdata(Path(bend_dir) / config['grid']['coords'])
    inds = np.array(inds)
    zk = zk[inds]
    grid = grid[:, inds]
    zk.flags.writeable = False
    grid.flags.writeable = False
    coords.flags.writeable = False
    return BendingMode(
        zk, config['zk']['R_outer'], config['zk'].get('R_inner', 0.0),
        coords[0], coords[1],
        grid[0], grid[1], grid[2], grid[3]
    )


@lru_cache(maxsize=4)
def load_bend(bend_dir, inds):
    with open(Path(bend_dir) / "bend.yaml") as f:
        config = yaml.safe_load(f)
    m1 = _load_mirror_bend(bend_dir, inds, config['M1'])
    m2 = _load_mirror_bend(bend_dir, inds, config['M2'])
    m3 = _load_mirror_bend(bend_dir, inds, config['M3'])
    return dict(M1=m1, M2=m2, M3=m3)


@lru_cache(maxsize=16*3)
def realize_bend(bend_dir, dof, inds, mirror):
    modes = load_bend(bend_dir, inds)[mirror]
    dof = np.array(dof)
    zk = galsim.zernike.Zernike(
        np.tensordot(dof, modes.zk, axes=1),
        R_outer=modes.R_outer,
        R_inner=modes.R_inner
    )
    z = np.tensordot(dof, modes.z, axes=1)
    dzdx = np.tensordot(dof, modes.dzdx, axes=1)
    dzdy = np.tensordot(dof, modes.dzdy, axes=1)
    d2zdxy = np.tensordot(dof, modes.d2zdxy, axes=1)
    grid = np.stack([z, dzdx, dzdy, d2zdxy])
    grid.flags.writeable = False
    return RealizedBend(zk, grid)


@lru_cache(maxsize=16)
def m2_gravity(fea_dir, zenith):
    data = _fits_cache(fea_dir, "M2_GT_FEA.fits.gz")
    if zenith is None:
        return np.zeros_like(data[0])

    zdz, hdz = data[0:2]

    out = zdz * (np.cos(zenith) - 1)
    out += hdz * np.sin(zenith)
    out *= 1e-6  # micron -> meters
    out.flags.writeable = False
    return out


@lru_cache(maxsize=16)
def m2_temperature(fea_dir, TzGrad, TrGrad):
    data = _fits_cache(fea_dir, "M2_GT_FEA.fits.gz")
    tzdz, trdz = data[2:4]

    out = TzGrad * tzdz
    out += TrGrad * trdz
    out *= 1e-6
    out.flags.writeable = False
    return out


@lru_cache(maxsize=16)
def LSSTCam_gravity(fea_dir, zenith, rotation):
    if zenith is None:
        return None

    camera_gravity_zk = {}
    cam_data = [
        ('L1S1', 'L1_entrance'),
        ('L1S2', 'L1_exit'),
        ('L2S1', 'L2_entrance'),
        ('L2S2', 'L2_exit'),
        ('L3S1', 'L3_entrance'),
        ('L3S2', 'L3_exit')
    ]
    for tname, bname in cam_data:
        data = _fits_cache(fea_dir, tname+"zer.fits.gz")
        grav_zk = data[0, 3:] * (np.cos(zenith) - 1)
        grav_zk += (
            data[1, 3:] * np.cos(rotation) +
            data[2, 3:] * np.sin(rotation)
        ) * np.sin(zenith)

        # remap Andy -> Noll Zernike indices
        zIdxMapping = [
            1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19, 18, 20,
            17, 21, 16, 25, 24, 26, 23, 27, 22, 28
        ]
        grav_zk = grav_zk[[x - 1 for x in zIdxMapping]]
        grav_zk *= 1e-3  # mm -> m
        # tsph -> batoid 0-index offset
        grav_zk = np.concatenate([[0], grav_zk])
        camera_gravity_zk[bname] = grav_zk
    return camera_gravity_zk


@lru_cache(maxsize=16)
def LSSTCam_temperature(fea_dir, TBulk):
    camera_temperature_zk = {}
    cam_data = [
        ('L1S1', 'L1_entrance'),
        ('L1S2', 'L1_exit'),
        ('L2S1', 'L2_entrance'),
        ('L2S2', 'L2_exit'),
        ('L3S1', 'L3_entrance'),
        ('L3S2', 'L3_exit')
    ]
    for tname, bname in cam_data:
        data = _fits_cache(fea_dir, tname+"zer.fits.gz")
        # subtract pre-compensated grav...
        TBulk = np.clip(
            TBulk,
            np.min(data[3:, 2])+0.001,
            np.max(data[3:, 2])-0.001
        )
        fidx = np.interp(TBulk, data[3:, 2], np.arange(len(data[3:, 2])))+3
        idx = int(np.floor(fidx))
        whi = fidx - idx
        wlo = 1 - whi
        temp_zk = wlo * data[idx, 3:] + whi * data[idx+1, 3:]

        # subtract reference temperature zk (0 deg C is idx=5)
        temp_zk -= data[5, 3:]

        # remap Andy -> Noll Zernike indices
        zIdxMapping = [
            1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19, 18, 20,
            17, 21, 16, 25, 24, 26, 23, 27, 22, 28
        ]
        temp_zk = temp_zk[[x - 1 for x in zIdxMapping]]
        temp_zk *= 1e-3  # mm -> m
        # tsph -> batoid 0-index offset
        temp_zk = np.concatenate([[0], temp_zk])
        camera_temperature_zk[bname] = temp_zk


class LSSTBuilder:
    def __init__(
        self,
        fiducial,
        fea_dir="fea_legacy",
        bend_dir="bend",
        use_m1m3_modes=None,
        use_m2_modes=None,
    ):
        """Create a Simony Survey Telescope with LSSTCam camera builder.

        Parameters
        ----------
        fiducial : batoid.Optic
            Optic before finite-element analysis (FEA) or active optics system
            (AOS) perturbations are applied.
        fea_dir : str
            Directory containing the FEA files.
        bend_dir : str
            Directory containing the bending mode files.
        use_m1m3_modes : list of int, optional
            List indicating the ordering of the bending modes to use for M1M3.
            For example, [9, 0, 4] would mean use the 9th, 0th, and 4th modes
            from the bending mode directory, but map these to the 10th, 11th,
            and 12th indices of the AOS degrees-of-freedom (The 0th through 9th
            indices are always reserved for the rigid body degrees of freedom of
            M2 and the camera).  If None, then use the modes in the order
            specified in the bending mode directory.
        use_m2_modes : list of int, optional
            List indicating the ordering of the bending modes to use for M2.
            Note that the M2 bending mode indices always come immediately after
            the M1M3 bending mode indices.  So if use_m1m3_modes is [9, 0, 4]
            and use_m2_modes is [4, 2], then the 10th, 11th, and 12th AOS
            degrees of freedom will be the 9th, 0th, and 4th M1M3 bending modes,
            and the 13th and 14th AOS degrees of freedom will be the 4th and 2nd
            M2 bending modes.  If None, then use the modes in the order
            specified in the bending mode directory.
        """
        # Number of FEA nodes and actuators is inferred from content of fea_dir.
        # Number of bending modes is inferred from content of bend_dir.
        # These can effect the sizes of intermediate numpy arrays, and also the
        # required inputs for the with_aos_dof and with_*_forces methods.

        self.fiducial = fiducial
        fea_dir = Path(fea_dir)
        bend_dir = Path(bend_dir)

        if not fea_dir.is_dir():
            # See if we can find the data in the batoid_rubin data directory.
            from . import datadir
            fea_dir = datadir / fea_dir
            if not fea_dir.is_dir():
                # See if we can download the data from Zenodo.
                from .data.download_rubin_data import (
                    download_rubin_data, zenodo_dois
                )
                if fea_dir.name in zenodo_dois:
                    args = namedtuple("Args", ["dataset", "outdir"])
                    args.dataset = fea_dir.parts[-1]
                    args.outdir = None
                    download_rubin_data(args)
                else:
                    raise ValueError("Cannot infer fea_dir.")

        if not bend_dir.is_dir():
            # See if we can find the data in the batoid_rubin data directory.
            from . import datadir
            bend_dir = datadir / bend_dir
            if not bend_dir.is_dir():
                # See if we can download the data from Zenodo.
                from .data.download_rubin_data import (
                    download_rubin_data, zenodo_dois
                )
                if bend_dir.name in zenodo_dois:
                    args = namedtuple("Args", ["dataset", "outdir"])
                    args.dataset = bend_dir.parts[-1]
                    args.outdir = None
                    download_rubin_data(args)
                else:
                    raise ValueError("Cannot infer bend_dir.")

        self.fea_dir = fea_dir
        self.bend_dir = bend_dir

        if use_m1m3_modes is None:
            with open(bend_dir / "bend.yaml") as f:
                config = yaml.safe_load(f)
                use_m1m3_modes = config['use_m1m3_modes']
        self.use_m1m3_modes = use_m1m3_modes
        self.m1m3_dof_indices = slice(10, 10+len(self.use_m1m3_modes))

        if use_m2_modes is None:
            with open(bend_dir / "bend.yaml") as f:
                config = yaml.safe_load(f)
                use_m2_modes = config['use_m2_modes']
        self.use_m2_modes = use_m2_modes
        self.m2_dof_indices = slice(
            self.m1m3_dof_indices.stop,
            self.m1m3_dof_indices.stop+len(self.use_m2_modes),
        )

        if 'LSST.LSSTCamera' in self.fiducial.itemDict:
            self.cam_name = 'LSSTCamera'
        elif 'ComCam.ComCam' in self.fiducial.itemDict:
            self.cam_name = 'ComCam.ComCam'
        else:
            raise ValueError("Unsupported optic")

        # "Input" variables.
        self.m1m3_zenith = None
        self.m1m3_TBulk = 0.0
        self.m1m3_TxGrad = 0.0
        self.m1m3_TyGrad = 0.0
        self.m1m3_TzGrad = 0.0
        self.m1m3_TrGrad = 0.0
        self.m1m3_lut_zenith = None
        self.m1m3_lut_error = 0.0
        self.m1m3_lut_seed = 1
        self.m1m3_extra_forces = np.zeros(256)

        self.m2_zenith = None
        self.m2_TzGrad = 0.0
        self.m2_TrGrad = 0.0

        self.camera_zenith = None
        self.camera_rotation = None
        self.camera_TBulk = None

        self.dof = np.zeros(50)
        self.extra_zk = None
        self.extra_zk_eps = None

    @attach_attr(
        _req_params={"zenith":galsim.Angle}
    )
    def with_m1m3_gravity(self, zenith):
        """Return new SSTBuilder that includes gravitational flexure of M1M3.

        Parameters
        ----------
        zenith : float
            Zenith angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with M1M3 gravitation flexure applied.
        """
        if isinstance(zenith, Real):
            zenith = zenith * galsim.radians

        ret = copy(self)
        ret.m1m3_zenith = zenith
        return ret

    @attach_attr(
        _req_params={
            "m1m3_TBulk":float,
        },
        _opt_params={
            "m1m3_TxGrad":float,
            "m1m3_TyGrad":float,
            "m1m3_TzGrad":float,
            "m1m3_TrGrad":float,
        }
    )
    def with_m1m3_temperature(
        self,
        m1m3_TBulk,
        m1m3_TxGrad=0.0,
        m1m3_TyGrad=0.0,
        m1m3_TzGrad=0.0,
        m1m3_TrGrad=0.0,
    ):
        """Return new SSTBuilder that includes temperature flexure of M1M3.

        Parameters
        ----------
        m1m3_TBulk : float
            Bulk temperature in C.
        m1m3_TxGrad : float, optional
            Temperature gradient in x direction in C / m (?)
        m1m3_TyGrad : float, optional
            Temperature gradient in y direction in C / m (?)
        m1m3_TzGrad : float, optional
            Temperature gradient in z direction in C / m (?)
        m1m3_TrGrad : float, optional
            Temperature gradient in r direction in C / m (?)

        Returns
        -------
        ret : SSTBuilder
            New builder with M1M3 temperature flexure applied.
        """
        ret = copy(self)
        ret.m1m3_TBulk = m1m3_TBulk
        ret.m1m3_TxGrad = m1m3_TxGrad
        ret.m1m3_TyGrad = m1m3_TyGrad
        ret.m1m3_TzGrad = m1m3_TzGrad
        ret.m1m3_TrGrad = m1m3_TrGrad
        return ret

    @attach_attr(
        _req_params={
            "zenith":galsim.Angle,
        },
        _opt_params={
            "error":float,
            "seed":int,
        }
    )
    def with_m1m3_lut(self, zenith, error=0.0, seed=1):
        """Return new SSTBuilder that includes LUT perturbations of M1M3.

        Parameters
        ----------
        zenith : float
            Zenith angle in radians
        error : float, optional
            Fractional error to apply to LUT forces.

        Returns
        -------
        ret : SSTBuilder
            New builder with M1M3 LUT applied.
        """
        if isinstance(zenith, Real):
            zenith = zenith * galsim.radians

        ret = copy(self)
        ret.m1m3_lut_zenith = zenith
        ret.m1m3_lut_error=error
        ret.m1m3_lut_seed=seed
        return ret

    @attach_attr(
        _req_params={"zenith":galsim.Angle},
    )
    def with_m2_gravity(self, zenith):
        """Return new SSTBuilder that includes gravitational flexure of M2.

        Parameters
        ----------
        zenith : float
            Zenith angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with M2 gravitation flexure applied.
        """
        if isinstance(zenith, Real):
            zenith = zenith * galsim.radians

        ret = copy(self)
        ret.m2_zenith = zenith
        return ret

    @attach_attr(
        _opt_params={
            "m2_TzGrad":float,
            "m2_TrGrad":float,
        }
    )
    def with_m2_temperature(
        self,
        m2_TzGrad=0.0,
        m2_TrGrad=0.0,
    ):
        """Return new SSTBuilder that includes temperature flexure of M2.

        Parameters
        ----------
        m2_TzGrad : float, optional
            Temperature gradient in z direction in C / m (?)
        m2_TrGrad : float, optional
            Temperature gradient in r direction in C / m (?)

        Returns
        -------
        ret : SSTBuilder
            New builder with M2 temperature flexure applied.
        """
        ret = copy(self)
        ret.m2_TzGrad = m2_TzGrad
        ret.m2_TrGrad = m2_TrGrad
        return ret

    @attach_attr(
        _req_params={
            "zenith":galsim.Angle,
            "rotation":galsim.Angle,
        },
    )
    def with_camera_gravity(self, zenith, rotation):
        """Return new SSTBuilder that includes gravitational flexure of camera.

        Parameters
        ----------
        zenith : float
            Zenith angle in radians
        rotation : float
            Rotation angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with camera gravitation flexure applied.
        """
        if isinstance(zenith, Real):
            zenith = zenith * galsim.radians
        if isinstance(rotation, Real):
            rotation = rotation * galsim.radians

        ret = copy(self)
        ret.camera_zenith = zenith
        ret.camera_rotation = rotation
        return ret

    @attach_attr(
        _req_params={
            "camera_TBulk":float,
        },
    )
    def with_camera_temperature(self, camera_TBulk):
        """Return new SSTBuilder that includes temperature flexure of camera.

        Parameters
        ----------
        camera_TBulk : float
            Camera temperature in C

        Returns
        -------
        ret : SSTBuilder
            New builder with camera temperature flexure applied.
        """
        ret = copy(self)
        ret.camera_TBulk = camera_TBulk
        return ret

    @attach_attr(
        _req_params={
            "dof":None,
        },
    )
    def with_aos_dof(self, dof):
        """Return new SSTBuilder that includes specified AOS degrees of freedom

        Parameters
        ----------
        dof : ndarray (50,)
            AOS degrees of freedom.
            0,1,2 are M2 z,x,y in micron
            3,4 are M2 rot around x, y in arcsec
            5,6,7 are camera z,x,y in micron
            8,9 are camera rot around x, y in arcsec
            The next len(use_m1m3_modes) are M1M3 bending modes in micron
            The next len(use_m2_modes) are M2 bending modes in micron

        Returns
        -------
        ret : SSTBuilder
            New builder with specified AOS DOF.
        """
        assert len(dof) == 10+len(self.use_m1m3_modes)+len(self.use_m2_modes)
        ret = copy(self)
        ret.dof = dof
        return ret

    @attach_attr(
        _opt_params={
            "dof":None,
            "dx":float,
            "dy":float,
            "dz":float,
            "rx":galsim.Angle,
            "ry":galsim.Angle,
        },
    )
    def with_m2_rigid(
        self, dof=None, dx=None, dy=None, dz=None, rx=None, ry=None
    ):
        """Return new SSTBuilder that includes specified M2 rigid body DOF

        Note that you cannot specify both dof and dx,dy,dz,rx,ry.
        Values specified here overwrite M2 rigid body values specified in
        with_aos_dof, but not other degrees of freedom.

        Parameters
        ----------
        dof : ndarray (5,), optional
            0,1,2 are M2 z,x,y in micron
            3,4 are M2 rot around x, y in arcsec
        dx, dy, dz : float
            M2 displacement in micron
        rx, ry : galsim.Angle
            M2 rotation around x, y

        Returns
        -------
        ret : SSTBuilder
            New builder with specified M2 rigid body DOF.
        """
        ret = copy(self)
        if (
            any([x is not None for x in [dx, dy, dz, rx, ry]])
            and dof is not None
        ):
            raise ValueError("Cannot specify both dof and dx,dy,dz,rx,ry")
        if dof is not None:
            ret.dof[:5] = dof
        else:
            if dz is not None:
                ret.dof[0] = dz
            if dx is not None:
                ret.dof[1] = dx
            if dy is not None:
                ret.dof[2] = dy
            if rx is not None:
                ret.dof[3] = rx.deg*3600
            if ry is not None:
                ret.dof[4] = ry.deg*3600
        return ret


    @attach_attr(
        _opt_params={
            "dof":None,
            "dx":float,
            "dy":float,
            "dz":float,
            "rx":galsim.Angle,
            "ry":galsim.Angle,
        },
    )
    def with_camera_rigid(
        self, dof=None, dx=None, dy=None, dz=None, rx=None, ry=None
    ):
        """Return new SSTBuilder that includes specified camera rigid body DOF

        Note that you cannot specify both dof and dx,dy,dz,rx,ry.
        Values specified here overwrite camera rigid body values specified in
        with_aos_dof, but not other degrees of freedom.

        Parameters
        ----------
        dof : ndarray (5,), optional
            0,1,2 are camera z,x,y in micron
            3,4 are camera rot around x, y in arcsec
        dx, dy, dz : float
            camera displacement in micron
        rx, ry : galsim.Angle
            camera rotation around x, y

        Returns
        -------
        ret : SSTBuilder
            New builder with specified camera rigid body DOF.
        """
        ret = copy(self)
        if (
            any([x is not None for x in [dx, dy, dz, rx, ry]])
            and dof is not None
        ):
            raise ValueError("Cannot specify both dof and dx,dy,dz,rx,ry")
        if dof is not None:
            ret.dof[5:10] = dof
        else:
            if dz is not None:
                ret.dof[5] = dz
            if dx is not None:
                ret.dof[6] = dx
            if dy is not None:
                ret.dof[7] = dy
            if rx is not None:
                ret.dof[8] = rx.deg*3600
            if ry is not None:
                ret.dof[9] = ry.deg*3600
        return ret

    @attach_attr(
        _req_params={
            "dof":None,
        },
    )
    def with_m1m3_bend(self, dof):
        """Return new SSTBuilder that includes specified M1M3 bending modes

        Parameters
        ----------
        dof : ndarray (len(use_m1m3_modes),)
            M1M3 bending modes in micron

        Returns
        -------
        ret : SSTBuilder
            New builder with specified M1M3 bending modes.
        """
        ret = copy(self)
        ret.dof[self.m1m3_dof_indices] = dof
        return ret

    @attach_attr(
        _req_params={
            "dof":None,
        },
    )
    def with_m2_bend(self, dof):
        """Return new SSTBuilder that includes specified M2 bending modes

        Parameters
        ----------
        dof : ndarray (len(use_m2_modes),)
            M2 bending modes in micron

        Returns
        -------
        ret : SSTBuilder
            New builder with specified M2 bending modes.
        """
        ret = copy(self)
        ret.dof[self.m2_dof_indices] = dof
        return ret

    @attach_attr(
        _req_params={
            "zk":None,
            "eps":float,
        },
    )
    def with_extra_zk(self, zk, eps):
        """Return new SSTBuilder that includes specified constant Zernike phase
        screen at entrance pupil.

        Parameters
        ----------
        zk : ndarray
            Zernike coefficients of phases to add at entrance pupil in meters.
        eps : float
            Use annular Zernikes with this fractional obscuration.

        Returns
        -------
        ret : SSTBuilder
            New builder with specified extra Zernike phases.
        """
        ret = copy(self)
        ret.extra_zk = zk
        ret.extra_zk_eps = eps
        return ret

    @attach_attr(
        _req_params={
            "forces":None,
        },
    )
    def with_m1m3_extra_forces(self, forces):
        """Return new SSTBuilder that includes specified M1M3 extra forces

        Parameters
        ----------
        forces : ndarray (256,)
            M1M3 extra forces in N

        Returns
        -------
        ret : SSTBuilder
            New builder with specified M1M3 extra forces.
        """
        # Todo: should be able to index these by actuator ID too.
        ret = copy(self)
        ret.m1m3_extra_forces = forces
        return ret

    def build(self):
        optic = self.fiducial
        optic = self._apply_phase(optic)
        optic = self._apply_rigid_body_perturbations(optic)
        optic = self._apply_M1M3_surface_perturbations(optic)
        optic = self._apply_M2_surface_perturbations(optic)
        optic = self._apply_camera_surface_perturbations(optic)
        return optic

    def _apply_phase(self, optic):
        if self.extra_zk is None:
            return optic
        optic = optic.withInsertedOptic(
            before="M1",
            item=batoid.OPDScreen(
                name='Screen',
                surface=batoid.Plane(),
                screen=batoid.Zernike(
                    self.extra_zk,
                    R_outer=4.18,
                    R_inner=self.extra_zk_eps*4.18
                ),
                coordSys=optic.stopSurface.coordSys,
                obscuration=optic['M1'].obscuration,
            )
        )
        return optic

    def _apply_rigid_body_perturbations(self, optic):
        dof = self.dof
        if np.any(dof[0:3]):
            optic = optic.withGloballyShiftedOptic(
                "M2",
                np.array([-dof[1], dof[2], -dof[0]])*1e-6
            )

        if np.any(dof[3:5]):
            rx = batoid.RotX(np.deg2rad(-dof[3]/3600))
            ry = batoid.RotY(np.deg2rad(dof[4]/3600))
            optic = optic.withLocallyRotatedOptic(
                "M2",
                rx @ ry
            )

        if np.any(dof[5:8]):
            optic = optic.withGloballyShiftedOptic(
                self.cam_name,
                np.array([-dof[6], dof[7], -dof[5]])*1e-6
            )

        if np.any(dof[8:10]):
            rx = batoid.RotX(np.deg2rad(-dof[8]/3600))
            ry = batoid.RotY(np.deg2rad(dof[9]/3600))
            optic = optic.withLocallyRotatedOptic(
                self.cam_name,
                rx @ ry
            )

        return optic

    def _apply_M1M3_surface_perturbations(self, optic):
        dof = self.dof
        # Collect gravity/temperature perturbations
        # be sure to make a copy here!
        m1m3_fea = np.array(m1m3_gravity(
            self.fea_dir, self.fiducial, self.m1m3_zenith
        ))
        m1m3_fea += m1m3_temperature(
            self.fea_dir,
            self.m1m3_TBulk,
            self.m1m3_TxGrad,
            self.m1m3_TyGrad,
            self.m1m3_TzGrad,
            self.m1m3_TrGrad
        )

        m1m3_forces = np.array(m1m3_lut(
            self.fea_dir,
            self.m1m3_lut_zenith,
            self.m1m3_lut_error,
            self.m1m3_lut_seed
        ))
        m1m3_forces += self.m1m3_extra_forces

        m1m3_fea += m1m3_force_to_surface(
            self.fea_dir,
            m1m3_forces
        )

        bx, by, idx1, idx3 = m1m3_fea_nodes(self.fea_dir)
        if np.any(m1m3_fea):
            # Decompose into independent annular Zernikes+grid on M1 and M3
            m1_fea = m1m3_fea[idx1]
            m3_fea = m1m3_fea[idx3]

            zBasis1 = galsim.zernike.zernikeBasis(
                28, bx[idx1], by[idx1], R_outer=4.18, R_inner=2.558
            )
            zBasis3 = galsim.zernike.zernikeBasis(
                28, bx[idx3], by[idx3], R_outer=2.508, R_inner=0.55
            )
            m1_zk, *_ = np.linalg.lstsq(zBasis1.T, m1_fea, rcond=None)
            m3_zk, *_ = np.linalg.lstsq(zBasis3.T, m3_fea, rcond=None)
            m1_zk = galsim.zernike.Zernike(m1_zk, R_outer=4.18, R_inner=2.558)
            m3_zk = galsim.zernike.Zernike(m3_zk, R_outer=2.508, R_inner=0.55)
            m1_fea -= m1_zk(bx[idx1], by[idx1])
            m3_fea -= m3_zk(bx[idx3], by[idx3])

            m1_grid = _node_to_grid(
                bx[idx1], by[idx1], m1_fea, m1m3_grid_xy(self.bend_dir)[0]
            )
            m3_grid = _node_to_grid(
                bx[idx3], by[idx3], m3_fea, m1m3_grid_xy(self.bend_dir)[1]
            )
        else:
            m1_nx = m1m3_grid_xy(self.bend_dir)[0].shape[1]
            m3_nx = m1m3_grid_xy(self.bend_dir)[1].shape[1]
            m1_grid = np.zeros((4, m1_nx, m1_nx))
            m3_grid = np.zeros((4, m3_nx, m3_nx))
            m1_zk = galsim.zernike.Zernike(
                np.zeros(29), R_outer=4.18, R_inner=2.558
            )
            m3_zk = galsim.zernike.Zernike(
                np.zeros(29), R_outer=2.508, R_inner=0.55
            )

        # Fold in M1M3 bending modes
        if np.any(dof[self.m1m3_dof_indices]):
            bend1 = realize_bend(
                self.bend_dir,
                tuple(dof[self.m1m3_dof_indices]),
                tuple(self.use_m1m3_modes),
                "M1"
            )
            bend3 = realize_bend(
                self.bend_dir,
                tuple(dof[self.m1m3_dof_indices]),
                tuple(self.use_m1m3_modes),
                "M3"
            )

            m1_zk += transform_zernike(bend1.zk, R_outer=4.18, R_inner=2.558)
            m3_zk += transform_zernike(bend3.zk, R_outer=2.508, R_inner=0.55)

            m1_grid += bend1.grid
            m3_grid += bend3.grid

        # Apply to M1
        components = []
        if np.any(m1_zk.coef):
            components.append(
                batoid.Zernike(
                    m1_zk.coef, m1_zk.R_outer, m1_zk.R_inner
                )
            )
        if np.any(m1_grid):
            components.append(
                batoid.Bicubic(
                    *m1m3_grid_xy(self.bend_dir)[0],
                    *m1_grid,
                    nanpolicy='zero'
                )
            )
        if components:
            if len(components) >= 2:
                perturbation = batoid.Sum(components)
            else:
                perturbation = components[0]
            optic = optic.withPerturbedSurface(
                'M1',
                perturbation
            )

        # Apply to M3
        components = []
        if np.any(m3_zk.coef):
            components.append(
                batoid.Zernike(
                    m3_zk.coef, m3_zk.R_outer, m3_zk.R_inner
                )
            )
        if np.any(m3_grid):
            components.append(
                batoid.Bicubic(
                    *m1m3_grid_xy(self.bend_dir)[1],
                    *m3_grid,
                    nanpolicy='zero'
                )
            )
        if components:
            if len(components) >= 2:
                perturbation = batoid.Sum(components)
            else:
                perturbation = components[0]
            optic = optic.withPerturbedSurface(
                'M3',
                perturbation
            )
        return optic

    def _apply_M2_surface_perturbations(self, optic):
        dof = self.dof
        # Collect gravity/temperature perturbations
        # be sure to make a copy here!
        m2_fea = np.array(m2_gravity(
            self.fea_dir, self.m2_zenith
        ))
        m2_fea += m2_temperature(
            self.fea_dir,
            self.m2_TzGrad,
            self.m2_TrGrad
        )
        # m2_fea += m2_lut(
        #     self.fea_dir,
        #     self.m2_lut_zenith,
        # )

        bx, by = m2_fea_nodes(self.fea_dir)
        if np.any(m2_fea):
            # Decompose into annular Zernikes+grid on M2
            zBasis2 = galsim.zernike.zernikeBasis(
                28, bx, by, R_outer=1.71, R_inner=0.9
            )
            m2_zk, *_ = np.linalg.lstsq(zBasis2.T, m2_fea, rcond=None)
            m2_zk = galsim.zernike.Zernike(m2_zk, R_outer=1.71, R_inner=0.9)
            m2_fea -= m2_zk(bx, by)

            m2_grid = _node_to_grid(
                bx, by, m2_fea, m2_grid_xy(self.bend_dir)
            )
        else:
            m2_nx = m2_grid_xy(self.bend_dir).shape[1]
            m2_grid = np.zeros((4, m2_nx, m2_nx))
            m2_zk = galsim.zernike.Zernike(
                np.zeros(29), R_outer=1.71, R_inner=0.9
            )

        # Fold in M2 bending modes
        if np.any(dof[self.m2_dof_indices]):
            bend2 = realize_bend(
                self.bend_dir,
                tuple(dof[self.m2_dof_indices]),
                tuple(self.use_m2_modes),
                "M2"
            )
            m2_zk += transform_zernike(bend2.zk, R_outer=1.71, R_inner=0.9)
            m2_grid += bend2.grid

        # Apply to M2
        components = []
        if np.any(m2_zk.coef):
            components.append(
                batoid.Zernike(
                    m2_zk.coef, m2_zk.R_outer, m2_zk.R_inner
                )
            )
        if np.any(m2_grid):
            components.append(
                batoid.Bicubic(
                    *m2_grid_xy(self.bend_dir),
                    *m2_grid,
                    nanpolicy='zero'
                )
            )
        if components:
            if len(components) >= 2:
                perturbation = batoid.Sum(components)
            else:
                perturbation = components[0]
            optic = optic.withPerturbedSurface(
                'M2',
                perturbation
            )
        return optic

    def _apply_camera_surface_perturbations(self, optic):
        if self.camera_zenith is None:
            zen = None
            rot = None
        else:
            zen = self.camera_zenith
            rot = self.camera_rotation
        TBulk = self.camera_TBulk

        for tname, bname, radius in [
            ('L1S1', 'L1_entrance', 0.775),
            ('L1S2', 'L1_exit', 0.775),
            ('L2S1', 'L2_entrance', 0.551),
            ('L2S2', 'L2_exit', 0.551),
            ('L3S1', 'L3_entrance', 0.361),
            ('L3S2', 'L3_exit', 0.361)
        ]:
            data = _fits_cache(self.fea_dir, tname+"zer.fits.gz")
            zk = np.zeros(28)
            if zen is not None:
                zk += data[0, 3:] * (np.cos(zen) - 1)
                zk += (
                    data[1, 3:] * np.cos(rot) +
                    data[2, 3:] * np.sin(rot)
                ) * np.sin(zen)

            if TBulk is not None:
                TBulk1 = np.clip(
                    TBulk,
                    np.min(data[3:, 2])+0.001,
                    np.max(data[3:, 2])-0.001
                )
                fidx = np.interp(
                    TBulk1, data[3:, 2], np.arange(len(data[3:, 2]))
                )+3
                idx = int(np.floor(fidx))
                whi = fidx - idx
                wlo = 1 - whi
                zk += wlo * data[idx, 3:] + whi * data[idx+1, 3:]

                # subtract reference temperature zk (0 deg C is idx=5)
                zk -= data[5, 3:]

            # remap Andy -> Noll Zernike indices
            zIdxMapping = [
                1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19, 18, 20,
                17, 21, 16, 25, 24, 26, 23, 27, 22, 28
            ]
            zk = zk[[x - 1 for x in zIdxMapping]]
            zk *= 1e-3  # mm -> m
            # tsph -> batoid 0-index offset
            zk = np.concatenate([[0], zk])

            # Now need to flip x and z for Zemax -> batoid
            for j in range(1, 29):
                n, m = galsim.zernike.noll_to_zern(j)
                if (n+(m>=0)) % 2 == 0:  # antisymmetric in x
                    zk[j] *= -1
            zk = -zk

            if np.any(zk):
                optic = optic.withPerturbedSurface(
                    bname,
                    batoid.Zernike(zk, R_outer=radius)
                )

        return optic

# Attach GalSim config attributes to use when constructing LSSTBuilder
# Don't actually require `fiducial`, since that's handled manually in the
# ImSim config.
LSSTBuilder._req_params = {}
LSSTBuilder._single_params = {}
LSSTBuilder._opt_params = {
    "fea_dir":str,
    "bend_dir":str,
    "use_m1m3_modes":None,
    "use_m2_modes":None,
}
# Ignore with_* args in the ImSim config dict
LSSTBuilder._ignore_params = {
    k[5:]:None for k in dir(LSSTBuilder) if k.startswith("with_")
}
