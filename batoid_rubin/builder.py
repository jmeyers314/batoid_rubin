from pathlib import Path
from collections import namedtuple
from copy import copy
from functools import lru_cache
from numbers import Real
import os

import astropy.io.fits as fits
import batoid
import galsim
import numpy as np
import yaml

from .utils import _node_to_grid, _fits_cache


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
def m1m3_gravity(fea_dir, optic, zenith_angle):
    zdata = _fits_cache(fea_dir, "M1M3_dxdydz_zenith.fits.gz")
    hdata = _fits_cache(fea_dir, "M1M3_dxdydz_horizon.fits.gz")

    if zenith_angle is None:
        return np.zeros_like(zdata[:,2])

    dxyz = (
        zdata * np.cos(zenith_angle) +
        hdata * np.sin(zenith_angle)
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
    tbdz, txdz, tydz, tzdz, trdz = _fits_cache(fea_dir, "M1M3_thermal_FEA.fits.gz")

    out = TBulk * tbdz
    out += TxGrad * txdz
    out += TyGrad * tydz
    out += TzGrad * tzdz
    out += TrGrad * trdz
    out *= 1e-6
    out.flags.writeable = False
    return out


@lru_cache(maxsize=16)
def m1m3_lut(fea_dir, zenith_angle, error, seed):
    if zenith_angle is None:
        return np.zeros(256)

    from scipy.interpolate import interp1d
    data = _fits_cache(fea_dir, "M1M3_LUT.fits.gz")

    LUT_force = interp1d(data[0], data[1:])(np.rad2deg(zenith_angle))

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
    u0 = zf * np.cos(zenith_angle)
    u0 += hf * np.sin(zenith_angle)

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


def _load_mirror_bend(bend_dir, config):
    zk = fits.getdata(Path(bend_dir) / config['zk']['file'])
    grid = fits.getdata(Path(bend_dir) / config['grid']['file'])
    coords = fits.getdata(Path(bend_dir) / config['grid']['coords'])
    zk.flags.writeable = False
    grid.flags.writeable = False
    coords.flags.writeable = False
    return BendingMode(
        zk, config['zk']['R_outer'], config['zk'].get('R_inner', 0.0),
        coords[0], coords[1],
        grid[0], grid[1], grid[2], grid[3]
    )


@lru_cache(maxsize=4)
def load_bend(bend_dir):
    with open(Path(bend_dir) / "bend.yaml") as f:
        config = yaml.safe_load(f)
    m1 = _load_mirror_bend(bend_dir, config['M1'])
    m2 = _load_mirror_bend(bend_dir, config['M2'])
    m3 = _load_mirror_bend(bend_dir, config['M3'])
    return m1, m2, m3


@lru_cache(maxsize=16*3)
def realize_bend(bend_dir, dof, i):
    modes = load_bend(bend_dir)[i]
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
def m2_gravity(fea_dir, zenith_angle):
    data = _fits_cache(fea_dir, "M2_GT_FEA.fits.gz")
    if zenith_angle is None:
        return np.zeros_like(data[0])

    zdz, hdz = data[0:2]

    out = zdz * (np.cos(zenith_angle) - 1)
    out += hdz * np.sin(zenith_angle)
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
def LSSTCam_gravity(fea_dir, zenith_angle, rotation_angle):
    if zenith_angle is None:
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
        grav_zk = data[0, 3:] * (np.cos(zenith_angle) - 1)
        grav_zk += (
            data[1, 3:] * np.cos(rotation_angle) +
            data[2, 3:] * np.sin(rotation_angle)
        ) * np.sin(zenith_angle)

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
    def __init__(self, fiducial, fea_dir="fea_legacy", bend_dir="bend_legacy"):
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
                raise ValueError("Cannot infer fea_dir.")

        if not bend_dir.is_dir():
            # See if we can find the data in the batoid_rubin data directory.
            from . import datadir
            bend_dir = datadir / bend_dir
            if not bend_dir.is_dir():
                raise ValueError("Cannot infer bend_dir.")


        self.fea_dir = fea_dir
        self.bend_dir = bend_dir

        if 'LSST.LSSTCamera' in self.fiducial.itemDict:
            self.cam_name = 'LSSTCamera'
        elif 'ComCam.ComCam' in self.fiducial.itemDict:
            self.cam_name = 'ComCam'
        else:
            raise ValueError("Unsupported optic")

        # "Input" variables.
        self.m1m3_zenith_angle = None
        self.m1m3_TBulk = 0.0
        self.m1m3_TxGrad = 0.0
        self.m1m3_TyGrad = 0.0
        self.m1m3_TzGrad = 0.0
        self.m1m3_TrGrad = 0.0
        self.m1m3_lut_zenith_angle = None
        self.m1m3_lut_error = 0.0
        self.m1m3_lut_seed = 1
        self.m1m3_extra_forces = np.zeros(256)

        self.m2_zenith_angle = None
        self.m2_TzGrad = 0.0
        self.m2_TrGrad = 0.0

        self.camera_zenith_angle = None
        self.camera_rotation_angle = None
        self.camera_TBulk = None

        self.dof = np.zeros(50)

    def with_m1m3_gravity(self, zenith_angle):
        """Return new SSTBuilder that includes gravitational flexure of M1M3.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with M1M3 gravitation flexure applied.
        """
        if isinstance(zenith_angle, Real):
            zenith_angle = zenith_angle * galsim.radians

        ret = copy(self)
        ret.m1m3_zenith_angle = zenith_angle
        return ret

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

    def with_m1m3_lut(self, zenith_angle, error=0.0, seed=1):
        """Return new SSTBuilder that includes LUT perturbations of M1M3.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle in radians
        error : float, optional
            Fractional error to apply to LUT forces.

        Returns
        -------
        ret : SSTBuilder
            New builder with M1M3 LUT applied.
        """
        if isinstance(zenith_angle, Real):
            zenith_angle = zenith_angle * galsim.radians

        ret = copy(self)
        ret.m1m3_lut_zenith_angle = zenith_angle
        ret.m1m3_lut_error=error
        ret.m1m3_lut_seed=seed
        return ret

    def with_m2_gravity(self, zenith_angle):
        """Return new SSTBuilder that includes gravitational flexure of M2.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with M2 gravitation flexure applied.
        """
        if isinstance(zenith_angle, Real):
            zenith_angle = zenith_angle * galsim.radians

        ret = copy(self)
        ret.m2_zenith_angle = zenith_angle
        return ret

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

    def with_camera_gravity(self, zenith_angle, rotation_angle):
        """Return new SSTBuilder that includes gravitational flexure of camera.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle in radians
        rotation_angle : float
            Rotation angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with camera gravitation flexure applied.
        """
        if isinstance(zenith_angle, Real):
            zenith_angle = zenith_angle * galsim.radians
        if isinstance(rotation_angle, Real):
            rotation_angle = rotation_angle * galsim.radians

        ret = copy(self)
        ret.camera_zenith_angle = zenith_angle
        ret.camera_rotation_angle = rotation_angle
        return ret

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
            10-29 are M1M3 bending modes in micron
            30-49 are M2 bending modes in micron

        Returns
        -------
        ret : SSTBuilder
            New builder with specified AOS DOF.
        """
        ret = copy(self)
        ret.dof = dof
        return ret

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
        optic = self._apply_rigid_body_perturbations(optic)
        optic = self._apply_M1M3_surface_perturbations(optic)
        optic = self._apply_M2_surface_perturbations(optic)
        optic = self._apply_camera_surface_perturbations(optic)
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
            self.fea_dir, self.fiducial, self.m1m3_zenith_angle.rad
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
            self.m1m3_lut_zenith_angle.rad,
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
            m1_zk = galsim.zernike.Zernike(np.zeros(29), R_outer=4.18, R_inner=2.558)
            m3_zk = galsim.zernike.Zernike(np.zeros(29), R_outer=2.508, R_inner=0.55)

        # Fold in M1M3 bending modes
        if np.any(dof[10:30]):
            bend1 = realize_bend(self.bend_dir, tuple(dof[10:30]), 0)
            bend3 = realize_bend(self.bend_dir, tuple(dof[10:30]), 2)

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
                    *m1_grid
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
                    *m3_grid
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
            self.fea_dir, self.m2_zenith_angle.rad
        ))
        m2_fea += m2_temperature(
            self.fea_dir,
            self.m2_TzGrad,
            self.m2_TrGrad
        )
        # m2_fea += m2_lut(
        #     self.fea_dir,
        #     self.m2_lut_zenith_angle.rad,
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
            m2_zk = galsim.zernike.Zernike(np.zeros(29), R_outer=1.71, R_inner=0.9)

        # Fold in M2 bending modes
        if np.any(dof[30:50]):
            bend2 = realize_bend(self.bend_dir, tuple(dof[30:50]), 1)
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
                    *m2_grid
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
        if self.camera_zenith_angle is None:
            zen = None
            rot = None
        else:
            zen = self.camera_zenith_angle.rad
            rot = self.camera_rotation_angle.rad
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
                ) * np.sin(zen  )

            if TBulk is not None:
                TBulk1 = np.clip(
                    TBulk,
                    np.min(data[3:, 2])+0.001,
                    np.max(data[3:, 2])-0.001
                )
                fidx = np.interp(TBulk1, data[3:, 2], np.arange(len(data[3:, 2])))+3
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
