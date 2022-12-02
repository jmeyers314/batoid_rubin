from collections import namedtuple
from copy import copy
from functools import cached_property, lru_cache
import os

import astropy.io.fits as fits
import batoid
import galsim
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial import Delaunay
import yaml


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
    out = np.zeros([4, ny, nx])
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
    return fits.getdata(
        os.path.join(
            datadir,
            fn
        )
    )


@lru_cache(maxsize=2)
def m1m3_fea_nodes(fea_dir):
    data = fits.getdata(os.path.join(fea_dir, "M1M3_1um_156_grid.fits.gz"))
    idx = data[:, 0]
    bx = data[:, 1]  # (5256,)
    by = data[:, 2]
    idx1 = (idx == 1)
    idx3 = (idx == 3)
    return bx, by, idx1, idx3


@lru_cache(maxsize=4)
def m1m3_grid_xy(bend_dir):
    m1_grid_xy = fits.getdata(os.path.join(bend_dir, "M1_bend_coords.fits.gz"))
    m3_grid_xy = fits.getdata(os.path.join(bend_dir, "M3_bend_coords.fits.gz"))
    return m1_grid_xy, m3_grid_xy


@lru_cache(maxsize=2)
def m2_fea_nodes(fea_dir):
    data = fits.getdata(os.path.join(fea_dir, "M2_1um_grid.fits.gz"))
    bx = -data[:, 1]  # meters
    by = data[:, 2]
    return bx, by


@lru_cache(maxsize=4)
def m2_grid_xy(bend_dir):
    return fits.getdata(os.path.join(bend_dir, "M2_bend_coords.fits.gz"))


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

    return dz


@lru_cache(maxsize=16)
def m1m3_temperature(fea_dir, TBulk, TxGrad, TyGrad, TzGrad, TrGrad):
    bx, by, *_ = m1m3_fea_nodes(fea_dir)
    normX = bx / 4.18
    normY = by / 4.18

    data = _fits_cache(fea_dir, "M1M3_thermal_FEA.fits.gz")
    delaunay = Delaunay(data[:, 0:2])
    tbdz = CloughTocher2DInterpolator(delaunay, data[:, 2])(normX, normY)
    txdz = CloughTocher2DInterpolator(delaunay, data[:, 3])(normX, normY)
    tydz = CloughTocher2DInterpolator(delaunay, data[:, 4])(normX, normY)
    tzdz = CloughTocher2DInterpolator(delaunay, data[:, 5])(normX, normY)
    trdz = CloughTocher2DInterpolator(delaunay, data[:, 6])(normX, normY)

    out = TBulk * tbdz
    out += TxGrad * txdz
    out += TyGrad * tydz
    out += TzGrad * tzdz
    out += TrGrad * trdz
    out *= 1e-6
    return out


@lru_cache(maxsize=16)
def m1m3_lut(fea_dir, zenith_angle, error, seed):
    G = _fits_cache(fea_dir, "M1M3_influence_256.fits.gz")

    if zenith_angle is None:
        return np.zeros(G.shape[1])

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
    return G.dot(LUT_force - u0)


BendingMode = namedtuple(
    "BendingMode",
    "zk R_outer R_inner x y z dzdx dzdy d2zdxy"
)


def _load_mirror_bend(bend_dir, config):
    zk = fits.getdata(os.path.join(bend_dir, config['zk']['file']))
    grid = fits.getdata(os.path.join(bend_dir, config['grid']['file']))
    coords = fits.getdata(os.path.join(bend_dir, config['grid']['coords']))
    return BendingMode(
        zk, config['zk']['R_outer'], config['zk']['R_inner'],
        coords[0], coords[1],
        grid[0], grid[1], grid[2], grid[3]
    )


@lru_cache(maxsize=4)
def load_bend(bend_dir):
    with open(os.path.join(bend_dir, "bend.yaml")) as f:
        config = yaml.safe_load(f)
    m1 = _load_mirror_bend(bend_dir, config['M1'])
    m2 = _load_mirror_bend(bend_dir, config['M2'])
    m3 = _load_mirror_bend(bend_dir, config['M3'])
    return m1, m2, m3


@lru_cache(maxsize=16*3)
def mirror_bend(bend_dir, dof, i):
    modes = load_bend(bend_dir)[i]
    dof = np.array(dof)
    return (
        np.tensordot(dof, modes.zk, axes=1),
        np.tensordot(dof, modes.z, axes=1),
        np.tensordot(dof, modes.dzdx, axes=1),
        np.tensordot(dof, modes.dzdy, axes=1),
        np.tensordot(dof, modes.d2zdxy, axes=1)
    )


@lru_cache(maxsize=16)
def m2_gravity(fea_dir, zenith_angle):
    data = _fits_cache(fea_dir, "M2_GT_grid.fits.gz")
    if zenith_angle is None:
        return np.zeros_like(zdz)

    zdz, hdz = data[0:2]

    out = zdz * (np.cos(zenith_angle) - 1)
    out += hdz * np.sin(zenith_angle)
    out *= 1e-6  # micron -> meters

    return out


@lru_cache(maxsize=16)
def m2_temperature(fea_dir, TzGrad, TrGrad):
    data = _fits_cache(fea_dir, "M2_GT_grid.fits.gz")
    tzdz, trdz = data[2:4]

    out = TzGrad * tzdz
    out += TrGrad * trdz
    out *= 1e-6

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
    def __init__(self, fiducial, fea_dir=None, bend_dir=None):
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
        self.fiducial = fiducial
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

    def build(self):
        optic = self.fiducial
        optic = self._apply_rigid_body_perturbations(optic)
        # optic = self._apply_surface_perturbations(optic)
        return optic

    def _apply_rigid_body_perturbations(self, optic):
        dof = self.dof
        if np.any(dof[0:3]):
            optic = optic.withGloballyShiftedOptic(
                "M2",
                np.array([dof[1], dof[2], -dof[0]])*1e-6
            )

        if np.any(dof[3:5]):
            rx = batoid.RotX(np.deg2rad(-dof[3]/3600))
            ry = batoid.RotY(np.deg2rad(-dof[4]/3600))
            optic = optic.withLocallyRotatedOptic(
                "M2",
                rx @ ry
            )

        if np.any(dof[5:8]):
            optic = optic.withGloballyShiftedOptic(
                self.cam_name,
                np.array([dof[6], dof[7], -dof[5]])*1e-6
            )

        if np.any(dof[8:10]):
            rx = batoid.RotX(np.deg2rad(-dof[8]/3600))
            ry = batoid.RotY(np.deg2rad(-dof[9]/3600))
            optic = optic.withLocallyRotatedOptic(
                self.cam_name,
                rx @ ry
            )

        return optic

    def _apply_surface_perturbations(self, optic):
        m1m3_fea = m1m3_gravity(
            self.fea_dir, self.fiducial, self.m1m3_zenith_angle
        )
        m1m3_fea += m1m3_temperature(
            self.fea_dir,
            self.m1m3_TBulk,
            self.m1m3_TxGrad,
            self.m1m3_TyGrad,
            self.m1m3_TzGrad,
            self.m1m3_TrGrad
        )
        m1m3_fea += m1m3_lut(
            self.fea_dir,
            self.m1m3_lut_zenith_angle,
            self.m1m3_lut_error,
            self.m1m3_lut_seed
        )

        if np.any(m1m3_fea):
            bx, by, idx1, idx3 = m1m3_fea_nodes(self.fea_dir)
            zBasis = galsim.zernike.zernikeBasis(
                28, -bx, by, R_outer=4.18
            )
            m1m3_zk, *_ = np.linalg.lstsq(zBasis.T, m1m3_fea, rcond=None)
            zern = galsim.zernike.Zernike(m1m3_zk, R_outer=4.18)
            m1m3_fea -= zern(-bx, by)

            m1_grid = _node_to_grid(
                bx[idx1], by[idx1], m1m3_fea[idx1], self.m1_grid_xy
            )

            m3_grid = _node_to_grid(
                bx[idx3], by[idx3], m1m3_fea[idx3], self.m3_grid_xy
            )
            m1_grid *= -1
            m3_grid *= -1
            m1m3_zk *= -1


            self._m1_fea_grid = m1_grid
            self._m3_fea_grid = m3_grid
            self._m1m3_fea_zk = m1m3_zk
        else:
            self._m1_fea_grid = None
            self._m3_fea_grid = None
            self._m1m3_fea_zk = None


    # def _consolidate_m1m3_fea(self):
    #     # Take
    #     #     _m1m3_fea_gravity,  _m1m3_fea_temperature, _m1m3_fea_lut
    #     # and set
    #     #     _m1_fea_grid, _m3_fea_grid, _m1m3_fea_zk
    #     if self._m1_fea_grid is not _Invalidated:
    #         return
    #     if (
    #         self._m1m3_fea_gravity is None
    #         and self._m1m3_fea_temperature is None
    #         and self._m1m3_fea_lut is None
    #     ):
    #         self._m1_fea_grid = None
    #         self._m3_fea_grid = None
    #         self._m1m3_fea_zk = None
    #         return
    #     m1m3_fea = np.zeros(5256)
    #     if self._m1m3_fea_gravity is not None:
    #         m1m3_fea = self._m1m3_fea_gravity
    #     if self._m1m3_fea_temperature is not None:
    #         m1m3_fea += self._m1m3_fea_temperature
    #     if self._m1m3_fea_lut is not None:
    #         m1m3_fea += self._m1m3_fea_lut

    #     if np.any(m1m3_fea):
    #         bx, by = self.m1m3_fea_xy
    #         idx1, idx3 = self.m1m3_fea_idx13
    #         zBasis = galsim.zernike.zernikeBasis(
    #             28, -bx, by, R_outer=4.18
    #         )
    #         m1m3_zk, *_ = np.linalg.lstsq(zBasis.T, m1m3_fea, rcond=None)
    #         zern = galsim.zernike.Zernike(m1m3_zk, R_outer=4.18)
    #         m1m3_fea -= zern(-bx, by)

    #         m1_grid = _node_to_grid(
    #             bx[idx1], by[idx1], m1m3_fea[idx1], self.m1_grid_xy
    #         )

    #         m3_grid = _node_to_grid(
    #             bx[idx3], by[idx3], m1m3_fea[idx3], self.m3_grid_xy
    #         )
    #         m1_grid *= -1
    #         m3_grid *= -1
    #         m1m3_zk *= -1
    #         self._m1_fea_grid = m1_grid
    #         self._m3_fea_grid = m3_grid
    #         self._m1m3_fea_zk = m1m3_zk
    #     else:
    #         self._m1_fea_grid = None
    #         self._m3_fea_grid = None
    #         self._m1m3_fea_zk = None

    # def _consolidate_m1_grid(self):
    #     # Take m1_fea_grid, m1_bend_grid and make m1_grid.
    #     if self._m1_grid is not _Invalidated:
    #         return
    #     if (
    #         self._m1_bend_grid is None
    #         and self._m1_fea_grid is None
    #     ):
    #         self._m1_grid = None
    #         return

    #     if self._m1_bend_grid is not None:
    #         m1_grid = self._m1_bend_grid
    #     else:
    #         m1_grid = np.zeros((4, 204, 204))
    #     if self._m1_fea_grid is not None:
    #         m1_grid += self._m1_fea_grid
    #     self._m1_grid = m1_grid

    # def _consolidate_m3_grid(self):
    #     # Take m3_fea_grid, m3_bend_grid and make m3_grid.
    #     if self._m3_grid is not _Invalidated:
    #         return
    #     if (
    #         self._m3_bend_grid is None
    #         and self._m3_fea_grid is None
    #     ):
    #         self._m3_grid = None
    #         return

    #     if self._m3_bend_grid is not None:
    #         m3_grid = self._m3_bend_grid
    #     else:
    #         m3_grid = np.zeros((4, 204, 204))
    #     if self._m3_fea_grid is not None:
    #         m3_grid += self._m3_fea_grid
    #     self._m3_grid = m3_grid

    # def _consolidate_m1m3_zk(self):
    #     if self._m1m3_zk is not _Invalidated:
    #         return
    #     if (
    #         self._m1m3_bend_zk is None
    #         and self._m1m3_fea_zk is None
    #     ):
    #         self._m1m3_zk = None
    #         return
    #     m1m3_zk = np.zeros(29)
    #     if self._m1m3_bend_zk is not None:
    #         m1m3_zk += self._m1m3_bend_zk
    #     if self._m1m3_fea_zk is not None:
    #         m1m3_zk += self._m1m3_fea_zk
    #     self._m1m3_zk = m1m3_zk

    # def _consolidate_m2_fea(self):
    #     if self._m2_fea_grid is not _Invalidated:
    #         return
    #     if (
    #         self._m2_fea_gravity is None
    #         and self._m2_fea_temperature is None
    #     ):
    #         self._m2_fea_grid = None
    #         self._m2_fea_zk = None
    #         return
    #     m2_fea = np.zeros(15984)
    #     if self._m2_fea_gravity is not None:
    #         m2_fea = self._m2_fea_gravity
    #     if self._m2_fea_temperature is not None:
    #         m2_fea += self._m2_fea_temperature

    #     if np.any(m2_fea):
    #         bx, by = self.m2_fea_xy
    #         zBasis = galsim.zernike.zernikeBasis(
    #             28, -bx, by, R_outer=1.71
    #         )
    #         m2_zk, *_ = np.linalg.lstsq(zBasis.T, m2_fea, rcond=None)
    #         zern = galsim.zernike.Zernike(m2_zk, R_outer=1.71)
    #         m2_fea -= zern(-bx, by)
    #         m2_grid = _node_to_grid(
    #             bx, by, m2_fea, self.m2_grid_xy
    #         )
    #         m2_grid *= -1
    #         m2_zk *= -1
    #         self._m2_fea_grid = m2_grid
    #         self._m2_fea_zk = m2_zk
    #     else:
    #         self._m2_fea_grid = None
    #         self._m2_fea_zk = None

    # def _consolidate_m2_grid(self):
    #     # Take m2_fea_grid, m2_bend_grid and make m2_grid.
    #     if self._m2_grid is not _Invalidated:
    #         return
    #     if (
    #         self._m2_bend_grid is None
    #         and self._m2_fea_grid is None
    #     ):
    #         self._m2_grid = None
    #         return

    #     if self._m2_bend_grid is not None:
    #         m2_grid = self._m2_bend_grid
    #     else:
    #         m2_grid = np.zeros((4, 204, 204))
    #     if self._m2_fea_grid is not None:
    #         m2_grid += self._m2_fea_grid
    #     self._m2_grid = m2_grid

    # def _consolidate_m2_zk(self):
    #     if self.m2_zk is not _Invalidated:
    #         return
    #     if (
    #         self._m2_bend_zk is None
    #         and self._m2_fea_zk is None
    #     ):
    #         self._m2_zk = None
    #         return
    #     m2_zk = np.zeros(29)
    #     if self._m2_bend_zk is not None:
    #         m2_zk += self._m2_bend_zk
    #     if self._m2_fea_zk is not None:
    #         m2_zk += self._m2_fea_zk
    #     self._m2_zk = m2_zk

    # def _consolidate_camera(self):
    #     if self._camera_zk is not _Invalidated:
    #         return
    #     if (
    #         self._camera_gravity_zk is None
    #         and self._camera_temperature_zk is None
    #     ):
    #         self._camera_zk = None
    #         return
    #     zk = {}
    #     for bname, radius in [
    #         ('L1_entrance', 0.775),
    #         ('L1_exit', 0.775),
    #         ('L2_entrance', 0.551),
    #         ('L2_exit', 0.551),
    #         ('L3_entrance', 0.361),
    #         ('L3_exit', 0.361),
    #     ]:
    #         zk[bname] = (np.zeros(29), radius)
    #         if self._camera_gravity_zk is not None:
    #             zk[bname][0][:] += self._camera_gravity_zk[bname]
    #         if self._camera_temperature_zk is not None:
    #             zk[bname][0][:] += self._camera_temperature_zk[bname]
    #     self._camera_zk = zk

    # def _apply_surface_perturbations(self, optic):
    #     # M1
    #     components = [optic['M1'].surface]
    #     if np.any(self._m1_grid):
    #         components.append(
    #             batoid.Bicubic(
    #                 *self.m1_grid_xy,
    #                 *self._m1_grid,
    #                 nanpolicy='zero'
    #             )
    #         )
    #     if np.any(self._m1m3_zk):
    #         components.append(
    #             batoid.Zernike(self._m1m3_zk, R_outer=4.18)
    #         )
    #     if len(components) > 1:
    #         optic = optic.withSurface('M1', batoid.Sum(components))

    #     # M3
    #     components = [optic['M3'].surface]
    #     if np.any(self._m3_grid):
    #         components.append(
    #             batoid.Bicubic(
    #                 *self.m3_grid_xy,
    #                 *self._m3_grid,
    #                 nanpolicy='zero'
    #             )
    #         )
    #     if np.any(self._m1m3_zk):
    #         components.append(
    #             # Note, using M1 R_outer here intentionally.
    #             batoid.Zernike(self._m1m3_zk, R_outer=4.18)
    #         )
    #     if len(components) > 1:
    #         optic = optic.withSurface('M3', batoid.Sum(components))

    #     # M2
    #     components = [optic['M2'].surface]
    #     if np.any(self._m2_grid):
    #         components.append(
    #             batoid.Bicubic(
    #                 *self.m2_grid_xy,
    #                 *self._m2_grid,
    #                 nanpolicy='zero'
    #             )
    #         )
    #     if np.any(self._m2_zk):
    #         components.append(
    #             batoid.Zernike(self._m2_zk, R_outer=1.71)
    #         )
    #     if len(components) > 1:
    #         optic = optic.withSurface('M2', batoid.Sum(components))

    #     # Camera
    #     if self._camera_zk is not None:
    #         for k, (zk, radius) in self._camera_zk.items():
    #             optic = optic.withSurface(
    #                 k,
    #                 batoid.Sum([
    #                     optic[k].surface,
    #                     batoid.Zernike(zk, R_outer=radius)
    #                 ])
    #             )

    #     return optic

    # def build(self):
    #     # Fill arrays (possibly with None if all dependencies are None)
    #     # We're manually traversing the DAG effectively
    #     self._compute_m1m3_gravity()
    #     self._compute_m1m3_temperature()
    #     self._compute_m1m3_lut()
    #     self._consolidate_m1m3_fea()
    #     self._compute_m1m3_bend()
    #     self._consolidate_m1_grid()
    #     self._consolidate_m3_grid()
    #     self._consolidate_m1m3_zk()

    #     self._compute_m2_gravity()
    #     self._compute_m2_temperature()
    #     self._consolidate_m2_fea()
    #     self._compute_m2_bend()
    #     self._consolidate_m2_grid()
    #     self._consolidate_m2_zk()

    #     self._compute_camera_gravity()
    #     self._compute_camera_temperature()
    #     self._consolidate_camera()

    #     optic = self.fiducial
    #     optic = self._apply_rigid_body_perturbations(optic)
    #     optic = self._apply_surface_perturbations(optic)
    #     return optic