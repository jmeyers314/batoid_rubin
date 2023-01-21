import galsim
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
import batoid
from .builder import LSSTBuilder


def get_zk(opd):
    xs = np.linspace(-1, 1, opd.shape[0])
    ys = np.linspace(-1, 1, opd.shape[1])
    xs, ys = np.meshgrid(xs, ys)
    w = ~opd.mask
    basis = galsim.zernike.zernikeBasis(22, xs[w], ys[w], R_inner=0.61)
    zk, *_ = np.linalg.lstsq(basis.T, opd[w], rcond=None)
    return zk


def sub_ptt(opd):
    xs = np.linspace(-1, 1, opd.shape[0])
    ys = np.linspace(-1, 1, opd.shape[1])
    xs, ys = np.meshgrid(xs, ys)
    zk = get_zk(opd)
    opd -= galsim.zernike.Zernike(zk[:4], R_inner=0.61)(xs, ys)
    return opd


from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:99AEFF-33.3:0025B3-50:000000-66.7:C7030D-100:FF9FA3
    (0.000, (0.600, 0.682, 1.000)),
    (0.333, (0.000, 0.145, 0.702)),
    (0.500, (0.000, 0.000, 0.000)),
    (0.667, (0.780, 0.012, 0.051)),
    (1.000, (1.000, 0.624, 0.639))))


class VisualizeApp:
    def __init__(self, fiducial, wavelength, fea_dir="fea_legacy", bend_dir="bend_legacy", debug=None):
        self.nupdate = 0
        if debug is None:
            debug = contextlib.redirect_stdout(None)
        self.debug = debug
        self.fiducial = fiducial
        self.wavelength = wavelength
        self.fea_dir = fea_dir
        self.bend_dir = bend_dir

        # widget variables
        self.thx = 0.0
        self.thy = 0.0
        self.zenith = 0.0
        self.rotation = 0.0
        self.nphot = int(2e5)
        self.fwhm = 0.7

        self.do_M1M3_gravity = False
        self.do_M1M3_temperature = False
        self.M1M3_TBulk = 0.0
        self.M1M3_TxGrad = 0.0
        self.M1M3_TyGrad = 0.0
        self.M1M3_TzGrad = 0.0
        self.M1M3_TrGrad = 0.0
        self.do_M1M3_LUT = False
        self.M1M3_LUT_error = 0.0
        self.M1M3_LUT_seed = 0

        self.do_M2_gravity = False
        self.do_M2_temperature = False
        self.M2_TzGrad = 0.0
        self.M2_TrGrad = 0.0

        self.do_camera_gravity = False
        self.do_camera_temperature = False
        self.camera_TBulk = 0.0

        self.dof_idx = 0
        self.dof_value = 0.0

        self.actuator_idx = 0
        self.actuator_value = 0.0

        # Output
        self.view = self._view()

        # Controls
        kwargs = {'layout':{'width':'180px'}}
        self.thx_control = ipywidgets.FloatText(value=0.0, step=0.25, description='Field x (deg)', **kwargs)
        self.thy_control = ipywidgets.FloatText(value=0.0, step=0.25, description='Field y (deg)', **kwargs)
        self.zenith_control = ipywidgets.BoundedFloatText(value=0.0, step=1.0, min=0.0, max=90.0, description="Zenith (deg)", **kwargs)
        self.rotation_control = ipywidgets.BoundedFloatText(value=0.0, step=1.0, min=-90.0, max=90.0, description="Rotation (deg)", **kwargs)
        self.nphot_control = ipywidgets.IntText(value=int(2e5), description='# phot')
        self.fwhm_control = ipywidgets.FloatText(value=0.7, description='seeing (arcsec)')
        self.point_control = ipywidgets.VBox([
            self.thx_control,
            self.thy_control,
            self.zenith_control,
            self.rotation_control,
            self.nphot_control,
            self.fwhm_control
        ])

        self.M1M3_gravity_checkbox = ipywidgets.Checkbox(value=False, description="Gravity")
        self.M1M3_temperature_checkbox = ipywidgets.Checkbox(value=False, description="Temperature")
        self.M1M3_TBulk_control = ipywidgets.FloatText(value=0.0, step=0.1, description="TBulk")
        self.M1M3_TxGrad_control = ipywidgets.FloatText(value=0.0, step=0.01, description="TxGrad")
        self.M1M3_TyGrad_control = ipywidgets.FloatText(value=0.0, step=0.01, description="TyGrad")
        self.M1M3_TzGrad_control = ipywidgets.FloatText(value=0.0, step=0.01, description="TzGrad")
        self.M1M3_TrGrad_control = ipywidgets.FloatText(value=0.0, step=0.01, description="TrGrad")
        self.M1M3_LUT_checkbox = ipywidgets.Checkbox(value=False, description="Apply LUT")
        self.M1M3_LUT_error_control = ipywidgets.FloatText(value=0.0, step=0.001, description="force error")
        self.M1M3_LUT_seed_control = ipywidgets.BoundedIntText(value=0, description="error seed", min=0)
        self.M1M3_control = ipywidgets.VBox([
            self.M1M3_gravity_checkbox,
            self.M1M3_temperature_checkbox,
            self.M1M3_TBulk_control,
            self.M1M3_TxGrad_control,
            self.M1M3_TyGrad_control,
            self.M1M3_TzGrad_control,
            self.M1M3_TrGrad_control,
            self.M1M3_LUT_checkbox,
            self.M1M3_LUT_error_control,
            self.M1M3_LUT_seed_control
        ])
        self.M2_gravity_checkbox = ipywidgets.Checkbox(value=False, description="Gravity")
        self.M2_temperature_checkbox = ipywidgets.Checkbox(value=False, description="Temperature")
        self.M2_TzGrad_control = ipywidgets.FloatText(value=0.0, step=0.01, description="TzGrad")
        self.M2_TrGrad_control = ipywidgets.FloatText(value=0.0, step=0.01, description="TrGrad")
        self.M2_control = ipywidgets.VBox([
            self.M2_gravity_checkbox,
            self.M2_temperature_checkbox,
            self.M2_TzGrad_control,
            self.M2_TrGrad_control
        ])

        self.camera_gravity_checkbox = ipywidgets.Checkbox(value=False, description="Gravity")
        self.camera_temperature_checkbox = ipywidgets.Checkbox(value=False, description="Temperature")
        self.camera_TBulk_control = ipywidgets.BoundedFloatText(value=0.0, step=0.1, description="TBulk", min=-10.0, max=25.0)
        self.camera_control = ipywidgets.VBox([
            self.camera_gravity_checkbox,
            self.camera_temperature_checkbox,
            self.camera_TBulk_control
        ])

        self.dof_idx_control = ipywidgets.BoundedIntText(value=0, min=0, max=49, description="index")
        self.dof_value_control = ipywidgets.FloatText(value=0.0, step=0.1, description="value (micron)")

        self.dof_control = ipywidgets.VBox([
            self.dof_idx_control,
            self.dof_value_control
        ])

        self.actuator_idx_control = ipywidgets.BoundedIntText(value=0, min=0, max=255, description="index")
        self.actuator_value_control = ipywidgets.FloatText(value=0.0, step=10.0, description="value (Newton)")


        self.actuator_control = ipywidgets.VBox([
            self.actuator_idx_control,
            self.actuator_value_control
        ])

        self.tab_control = ipywidgets.Tab()
        self.tab_control.children = [
            self.point_control,
            self.M1M3_control,
            self.M2_control,
            self.camera_control,
            self.dof_control,
            self.actuator_control,
        ]

        # Lame interface, but works
        self.tab_control.set_title(0, 'Point')
        self.tab_control.set_title(1, 'M1M3')
        self.tab_control.set_title(2, 'M2')
        self.tab_control.set_title(3, 'Cam')
        self.tab_control.set_title(4, 'DOF')
        self.tab_control.set_title(5, 'Force')
        self.tab_control.layout.width='340px'

        self.thx_control.observe(self.handle_thx, 'value')
        self.thy_control.observe(self.handle_thy, 'value')
        self.zenith_control.observe(self.handle_zenith, 'value')
        self.rotation_control.observe(self.handle_rotation, 'value')
        self.nphot_control.observe(self.handle_nphot, 'value')
        self.fwhm_control.observe(self.handle_fwhm, 'value')

        self.M1M3_gravity_checkbox.observe(self.handle_M1M3_grav_check, 'value')
        self.M1M3_temperature_checkbox.observe(self.handle_M1M3_temp_check, 'value')
        self.M1M3_TBulk_control.observe(self.handle_M1M3_TBulk, 'value')
        self.M1M3_TxGrad_control.observe(self.handle_M1M3_TxGrad, 'value')
        self.M1M3_TyGrad_control.observe(self.handle_M1M3_TyGrad, 'value')
        self.M1M3_TzGrad_control.observe(self.handle_M1M3_TzGrad, 'value')
        self.M1M3_TrGrad_control.observe(self.handle_M1M3_TrGrad, 'value')

        self.M1M3_LUT_checkbox.observe(self.handle_M1M3_LUT_check, 'value')
        self.M1M3_LUT_error_control.observe(self.handle_M1M3_LUT_error, 'value')
        self.M1M3_LUT_seed_control.observe(self.handle_M1M3_LUT_seed, 'value')

        self.M2_gravity_checkbox.observe(self.handle_M2_grav_check, 'value')
        self.M2_temperature_checkbox.observe(self.handle_M2_temp_check, 'value')
        self.M2_TzGrad_control.observe(self.handle_M2_TzGrad, 'value')
        self.M2_TrGrad_control.observe(self.handle_M2_TrGrad, 'value')

        self.camera_gravity_checkbox.observe(self.handle_camera_grav_check, 'value')
        self.camera_temperature_checkbox.observe(self.handle_camera_temp_check, 'value')
        self.camera_TBulk_control.observe(self.handle_camera_TBulk, 'value')

        self.dof_idx_control.observe(self.handle_dof_idx, 'value')
        self.dof_value_control.observe(self.handle_dof_value, 'value')

        self.actuator_idx_control.observe(self.handle_actuator_idx, 'value')
        self.actuator_value_control.observe(self.handle_actuator_value, 'value')

        self.update()

    def handle_thx(self, change):
        self.thx = change['new']
        self.update()

    def handle_thy(self, change):
        self.thy = change['new']
        self.update()

    def handle_zenith(self, change):
        self.zenith = change['new']
        self.update()

    def handle_rotation(self, change):
        self.rotation = change['new']
        self.update()

    def handle_nphot(self, change):
        self.nphot = change['new']
        self.update()

    def handle_fwhm(self, change):
        self.fwhm = change['new']
        self.update()

    def handle_M1M3_grav_check(self, change):
        self.do_M1M3_gravity = not self.do_M1M3_gravity
        self.update()

    def handle_M1M3_temp_check(self, change):
        self.do_M1M3_temperature = not self.do_M1M3_temperature
        self.update()

    def handle_M1M3_TBulk(self, change):
        self.M1M3_TBulk = change['new']
        self.update()

    def handle_M1M3_TxGrad(self, change):
        self.M1M3_TxGrad = change['new']
        self.update()

    def handle_M1M3_TyGrad(self, change):
        self.M1M3_TyGrad = change['new']
        self.update()

    def handle_M1M3_TzGrad(self, change):
        self.M1M3_TzGrad = change['new']
        self.update()

    def handle_M1M3_TrGrad(self, change):
        self.M1M3_TrGrad = change['new']
        self.update()

    def handle_M1M3_LUT_check(self, change):
        self.do_M1M3_LUT = not self.do_M1M3_LUT
        self.update()

    def handle_M1M3_LUT_error(self, change):
        self.M1M3_LUT_error = change['new']
        self.update()

    def handle_M1M3_LUT_seed(self, change):
        self.M1M3_LUT_seed = change['new']
        self.update()

    def handle_M2_grav_check(self, change):
        self.do_M2_gravity = not self.do_M2_gravity
        self.update()

    def handle_M2_temp_check(self, change):
        self.do_M2_temperature = not self.do_M2_temperature
        self.update()

    def handle_M2_TzGrad(self, change):
        self.M2_TzGrad = change['new']
        self.update()

    def handle_M2_TrGrad(self, change):
        self.M2_TrGrad = change['new']
        self.update()

    def handle_camera_grav_check(self, change):
        self.do_camera_gravity = not self.do_camera_gravity
        self.update()

    def handle_camera_temp_check(self, change):
        self.do_camera_temperature = not self.do_camera_temperature
        self.update()

    def handle_camera_TBulk(self, change):
        self.camera_TBulk = change['new']
        self.update()

    def handle_dof_idx(self, change):
        self.dof_idx = change['new']
        self.update()

    def handle_dof_value(self, change):
        self.dof_value = change['new']
        self.update()

    def handle_actuator_idx(self, change):
        self.actuator_idx = change['new']
        self.update()

    def handle_actuator_value(self, change):
        self.actuator_value = change['new']
        self.update()

    def _view(self):
        self._fig, axes = plt.subplots(
            nrows=2, ncols=3,
            figsize=(6, 6),
            dpi=100,
            facecolor='#111111'
        )
        self._canvas = self._fig.canvas
        self._canvas.header_visible=False

        # Top Left/Center = wavefronts
        self.wfp_ax = axes[0,0]
        self.wfp_img = self.wfp_ax.imshow(
            np.ones((255, 255)),
            vmin=-1.0, vmax=1.0,
            cmap='seismic',
            extent=np.r_[1, -1, 1, -1]*4.18
        )
        self.wfp_ax.set_title("Wavefront", color='w')

        self.wfr_ax = axes[0,1]
        self.wfr_img = self.wfr_ax.imshow(
            np.zeros((255, 255)),
            vmin=-1.0, vmax=1.0,
            cmap='seismic',
            extent=np.r_[-1, 1, -1, 1]*4.18
        )
        self.wfr_ax.set_title("Wavefront residual", color='w')

        # Bottom Left/Center = donuts
        self.intra_ax = axes[1,0]
        self.intra_img = self.intra_ax.imshow(
            np.zeros((255, 255)),
            vmin=0.0, vmax=1.0,
            cmap='plasma',
            extent=np.r_[-1, 1, -1, 1]*0.2*255/2
        )
        self.intra_ax.set_title("Intra donut", color='w')

        self.extra_ax = axes[1,1]
        self.extra_img = self.extra_ax.imshow(
            np.zeros((255, 255)),
            vmin=0.0, vmax=1.0,
            cmap='plasma',
            extent=np.r_[-1, 1, -1, 1]*0.2*255/2
        )
        self.extra_ax.set_title("Extra donut", color='w')

        # Top Right = M1M3
        self.m1m3_ax = axes[0,2]
        self.m1m3_ax.set_facecolor('#111111')
        x = np.linspace(-4.18, 4.18, 255)
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        self.m1m3_mask = (
            rr < 0.55
        ) | (
            (rr > 2.508) & (rr < 2.558)
        ) | (
            rr > 4.18
        )
        m1m3_arr = np.ma.array(
            np.zeros((255, 255)),
            mask=self.m1m3_mask
        )
        self.m1m3_img = self.m1m3_ax.imshow(
            m1m3_arr,
            vmin=-1.0, vmax=1.0,
            cmap=cmap,
            extent=np.r_[1, -1, 1, -1]*4.18
        )
        self.m1m3_ax.set_title("M1M3", color='w')

        # Bottom Right = M2
        self.m2_ax = axes[1,2]
        self.m2_ax.set_facecolor('#111111')
        x = np.linspace(-1.71, 1.71, 255)
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        self.m2_mask = (rr < 0.9) | (rr > 1.71)
        m2_arr = np.ma.array(
            np.zeros((255, 255)),
            mask=self.m2_mask
        )
        self.m2_img = self.m2_ax.imshow(
            m2_arr,
            vmin=-1.0, vmax=1.0,
            cmap=cmap,
            extent=np.r_[1, -1, 1, -1]*1.71
        )
        self.m2_ax.set_title("M2", color='w')

        self._fig.tight_layout()
        out = ipywidgets.Output()
        with out:
            plt.show(self._fig)
        return out

    def update(self):
        self.nupdate += 1
        # Get the telescopes
        builder = LSSTBuilder(self.fiducial, self.fea_dir, self.bend_dir)
        tel0 = builder.build()
        tel0 = tel0.withLocallyRotatedOptic(
            "LSSTCamera", batoid.RotZ(np.deg2rad(self.rotation))
        )

        if self.do_M1M3_gravity:
            builder = builder.with_m1m3_gravity(
                np.deg2rad(self.zenith)
            )
        if self.do_M1M3_temperature:
            builder = builder.with_m1m3_temperature(
                self.M1M3_TBulk,
                self.M1M3_TxGrad,
                self.M1M3_TyGrad,
                self.M1M3_TzGrad,
                self.M1M3_TrGrad
            )
        if self.do_M1M3_LUT:
            builder = builder.with_m1m3_lut(
                np.deg2rad(self.zenith),
                self.M1M3_LUT_error,
                self.M1M3_LUT_seed
            )

        if self.do_M2_gravity:
            builder = builder.with_m2_gravity(
                np.deg2rad(self.zenith)
            )
        if self.do_M2_temperature:
            builder = builder.with_m2_temperature(
                self.M2_TzGrad, self.M2_TrGrad
            )

        if self.do_camera_gravity:
            builder = builder.with_camera_gravity(
                np.deg2rad(self.zenith),
                np.deg2rad(self.rotation)
            )
        if self.do_camera_temperature:
            builder = builder.with_camera_temperature(
                self.camera_TBulk
            )

        if self.dof_value != 0.0:
            dof = np.zeros(50)
            dof[self.dof_idx] = self.dof_value
            builder = builder.with_aos_dof(dof)

        if self.actuator_value != 0.0:
            forces = np.zeros(256)
            forces[self.actuator_idx] = self.actuator_value
            builder = builder.with_m1m3_extra_forces(forces)

        tel1 = builder.build()
        tel1 = tel1.withLocallyRotatedOptic(
            "LSSTCamera", batoid.RotZ(np.deg2rad(self.rotation))
        )

        # Update wavefront
        wf0 = sub_ptt(batoid.wavefront(
            tel0,
            np.deg2rad(self.thx), np.deg2rad(self.thy),
            self.wavelength,
            nx=255
        ).array)
        wf1 = sub_ptt(batoid.wavefront(
            tel1,
            np.deg2rad(self.thx), np.deg2rad(self.thy),
            self.wavelength,
            nx=255
        ).array)
        self.wfr_img.set_array(wf1-wf0)
        self.wfp_img.set_array(wf1)

        # Make some donuts
        intra = tel1.withGloballyShiftedOptic("Detector", (0, 0, -0.0015))
        extra = tel1.withGloballyShiftedOptic("Detector", (0, 0, +0.0015))
        rv = batoid.RayVector.asPolar(
            optic=tel1, wavelength=self.wavelength,
            nrandom=self.nphot,
            theta_x=np.deg2rad(self.thx),
            theta_y=np.deg2rad(self.thy)
        )
        rvi = intra.trace(rv.copy())
        rve = extra.trace(rv)

        rvi.x[:] -= np.mean(rvi.x[~rvi.vignetted])
        rvi.y[:] -= np.mean(rvi.y[~rvi.vignetted])
        rve.x[:] -= np.mean(rve.x[~rve.vignetted])
        rve.y[:] -= np.mean(rve.y[~rve.vignetted])

        # Convolve in a Gaussian
        scale = 10e-6 * self.fwhm/2.35/0.2
        rvi.x[:] += np.random.normal(scale=scale, size=len(rvi))
        rvi.y[:] += np.random.normal(scale=scale, size=len(rvi))
        rve.x[:] += np.random.normal(scale=scale, size=len(rve))
        rve.y[:] += np.random.normal(scale=scale, size=len(rve))

        # Bin rays
        di, _, _ = np.histogram2d(
            rvi.y[~rvi.vignetted], rvi.x[~rvi.vignetted], bins=255,
            range=[[-127*10e-6, 127*10e-6], [-127*10e-6, 127*10e-6]]
        )
        de, _, _ = np.histogram2d(
            rve.y[~rve.vignetted], rve.x[~rve.vignetted], bins=255,
            range=[[-127*10e-6, 127*10e-6], [-127*10e-6, 127*10e-6]]
        )

        self.intra_img.set_array(di/np.max(di))
        self.extra_img.set_array(de/np.max(de))

        # Update M1M3 shape
        x = np.linspace(-4.18, 4.18, 255)
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        w1 = (rr >= 2.558) & (rr <= 4.18)
        w3 = (rr >= 0.55) & (rr <= 2.508)
        sag = np.zeros((255, 255))
        sag[w1] = tel1['M1'].surface.sag(xx[w1], yy[w1])
        sag[w1] -= tel0['M1'].surface.sag(xx[w1], yy[w1])
        sag[w3] = tel1['M3'].surface.sag(xx[w3], yy[w3])
        sag[w3] -= tel0['M3'].surface.sag(xx[w3], yy[w3])
        sag *= 1e6
        m1m3_arr = np.ma.array(sag, mask=self.m1m3_mask)
        self.m1m3_img.set_array(m1m3_arr)

        # Update M2 shape
        x = np.linspace(-1.71, 1.71, 255)
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        w = (rr >= 0.9) & (rr <= 1.71)
        sag = np.zeros((255, 255))
        sag[w] = tel1['M2'].surface.sag(xx[w], yy[w])
        sag[w] -= tel0['M2'].surface.sag(xx[w], yy[w])
        sag *= 1e6
        m2_arr = np.ma.masked_array(sag, mask=self.m2_mask)
        self.m2_img.set_array(m2_arr)

        self._canvas.draw()

    def display(self):
        from IPython.display import display
        self.app = ipywidgets.HBox([
            self.view,
            self.tab_control
        ])

        display(self.app)
        self.update()  # not sure why this is needed, but works with this here.
