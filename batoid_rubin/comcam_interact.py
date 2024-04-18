import contextlib
from functools import cached_property
from pathlib import Path
import yaml

import batoid
import batoid_rubin
import danish
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares


class Sensor:
    def __init__(self, name, thx, thy, nphot, img, fiducial, focusz=0.0, wavelength=500e-9):
        self.name = name
        self.thx = thx
        self.thy = thy
        self.nphot = nphot
        self.img = img
        self.fiducial = fiducial
        self.focusz = focusz
        self.wavelength = wavelength
        self.bins = img.get_array().shape[0]
        bo2 = self.bins//2
        self.range = [[-bo2*10e-6, bo2*10e-6], [-bo2*10e-6, bo2*10e-6]]

        telescope = self.fiducial.withGloballyShiftedOptic(
            "Detector", [0, 0, focusz]
        )

        self.z_ref = batoid.zernikeTA(
            telescope, np.deg2rad(self.thx), np.deg2rad(self.thy),
            self.wavelength,
            nrad=20, naz=120,
            reference='chief',
            jmax=66, eps=0.61, focal_length=10.31
        )*self.wavelength  # meters


    def draw(self, telescope, seeing, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)

        telescope = telescope.withGloballyShiftedOptic(
            "Detector", [0, 0, self.focusz]
        )

        rv = batoid.RayVector.asPolar(
            optic=telescope, wavelength=self.wavelength,
            nrandom=self.nphot, rng=rng,
            theta_x=np.deg2rad(self.thx), theta_y=np.deg2rad(self.thy)
        )
        telescope.trace(rv)

        rv.x[:] -= np.mean(rv.x[~rv.vignetted])
        rv.y[:] -= np.mean(rv.y[~rv.vignetted])

        # Convolve in a Gaussian
        scale = 10e-6 * seeing/2.35/0.2
        rv.x[:] += rng.normal(scale=scale, size=len(rv))
        rv.y[:] += rng.normal(scale=scale, size=len(rv))

        # Bin rays
        psf, _, _ = np.histogram2d(
            rv.y[~rv.vignetted], rv.x[~rv.vignetted], bins=self.bins,
            range=self.range
        )
        self.img.set_array(psf/np.max(psf))


class ComCamAOS:
    def __init__(
        self,
        debug=None,
        control_log=None,
        rng=None,
        nthread=4,
        randomized_dof=list(range(10)),
        controlled_dof=list(range(10)),
        controlled_truncated_dof=[0,1,2,6,7,8,9],
        dz_terms = ((1, 4), (2, 4), (3, 4), (2, 5), (3, 5), (2, 6), (3, 6), (1, 7), (1, 8)),
        alpha=1e-12
    ):
        batoid._batoid.set_nthreads(nthread)

        if debug is None:
            debug = contextlib.redirect_stdout(None)
        self.debug = debug
        if control_log is None:
            control_log = contextlib.redirect_stdout(None)
        self.control_log = control_log
        if isinstance(self.debug, ipywidgets.Output):
            self.debug.clear_output()
        if isinstance(self.control_log, ipywidgets.Output):
            self.control_log.clear_output()

        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng
        self.offset_rng = np.random.default_rng(rng.integers(2**63))

        self.randomized_dof = randomized_dof
        self.controlled_dof = controlled_dof
        self.controlled_truncated_dof = controlled_truncated_dof

        self.dz_terms = dz_terms
        self.alpha = alpha

        self.fiducial = batoid.Optic.fromYaml("ComCam_r.yaml")
        self.builder = batoid_rubin.LSSTBuilder(self.fiducial)
        self.wavelength = 500e-9

        # widget variables
        self.m2_dz = 0.0
        self.m2_dx = 0.0
        self.m2_dy = 0.0
        self.m2_Rx = 0.0
        self.m2_Ry = 0.0

        self.cam_dz = 0.0
        self.cam_dx = 0.0
        self.cam_dy = 0.0
        self.cam_Rx = 0.0
        self.cam_Ry = 0.0

        self.offsets = np.zeros_like(self.randomized_dof)
        self.text = ""
        self._n_iter = 0

        # Controls
        kwargs = {'layout':{'width':'180px'}, 'style':{'description_width':'initial'}}
        self.m2_dz_control = ipywidgets.FloatText(value=self.m2_dz, description="M2 dz (µm)", step=10, **kwargs)
        self.m2_dx_control = ipywidgets.FloatText(value=self.m2_dx, description="M2 dx (µm)", step=500, **kwargs)
        self.m2_dy_control = ipywidgets.FloatText(value=self.m2_dy, description="M2 dy (µm)", step=500, **kwargs)
        self.m2_Rx_control = ipywidgets.FloatText(value=self.m2_Rx, description="M2 Rx (arcsec)", step=10, **kwargs)
        self.m2_Ry_control = ipywidgets.FloatText(value=self.m2_Ry, description="M2 Ry (arcsec)", step=10, **kwargs)
        self.cam_dz_control = ipywidgets.FloatText(value=self.cam_dz, description="Cam dz (µm)", step=10, **kwargs)
        self.cam_dx_control = ipywidgets.FloatText(value=self.cam_dx, description="Cam dx (µm)", step=2000, **kwargs)
        self.cam_dy_control = ipywidgets.FloatText(value=self.cam_dy, description="Cam dy (µm)", step=2000, **kwargs)
        self.cam_Rx_control = ipywidgets.FloatText(value=self.cam_Rx, description="Cam Rx (arcsec)", step=10, **kwargs)
        self.cam_Ry_control = ipywidgets.FloatText(value=self.cam_Ry, description="Cam Ry (arcsec)", step=10, **kwargs)
        self.zero_control = ipywidgets.Button(description="Zero")
        self.randomize_control = ipywidgets.Button(description="Randomize")
        self.reveal_control = ipywidgets.Button(description="Reveal")
        self.solve_control = ipywidgets.Button(description="Solve")
        self.control_truncated_control = ipywidgets.Button(description="Control w/ Trunc")
        self.control_penalty_control = ipywidgets.Button(description="Control w/ Penalty")

        self.hex_controls = ipywidgets.VBox([
            self.m2_dz_control, self.m2_dx_control, self.m2_dy_control,
            self.m2_Rx_control, self.m2_Ry_control,
            self.cam_dz_control, self.cam_dx_control, self.cam_dy_control,
            self.cam_Rx_control, self.cam_Ry_control,
        ])

        self.m1m3_vals = [0.0]*20
        self.m1m3_controls = [
            ipywidgets.FloatText(value=self.m1m3_vals[i], description=f"Mode {i} (µm)", step=0.25, **kwargs)
            for i in range(20)
        ]
        self.m1m3_control_box1 = ipywidgets.VBox(self.m1m3_controls[:10])
        self.m1m3_control_box2 = ipywidgets.VBox(self.m1m3_controls[10:])

        self.m2_vals = [0.0]*20
        self.m2_controls = [
            ipywidgets.FloatText(value=self.m2_vals[i], description=f"Mode {i} (µm)", step=0.25, **kwargs)
            for i in range(20)
        ]
        self.m2_control_box1 = ipywidgets.VBox(self.m2_controls[:10])
        self.m2_control_box2 = ipywidgets.VBox(self.m2_controls[10:])

        self.component_controls = ipywidgets.Tab(layout=ipywidgets.Layout(min_width="200pt", height="290pt"))
        self.component_controls.children = [
            self.hex_controls,
            self.m1m3_control_box1, self.m1m3_control_box2,
            self.m2_control_box1, self.m2_control_box2,
        ]
        self.component_controls.set_title(0, "Hex")
        self.component_controls.set_title(1, "M1M3a")
        self.component_controls.set_title(2, "M1M3b")
        self.component_controls.set_title(3, "M2a")
        self.component_controls.set_title(4, "M2b")

        self.controls = ipywidgets.VBox([
            self.component_controls,
            ipywidgets.VBox([
                self.zero_control, self.randomize_control,
                self.reveal_control, self.solve_control,
                self.control_truncated_control, self.control_penalty_control
            ])
        ])

        # Observers
        self.m2_dz_control.observe(lambda change: self.handle_event(change, 'm2_dz'), 'value')
        self.m2_dx_control.observe(lambda change: self.handle_event(change, 'm2_dx'), 'value')
        self.m2_dy_control.observe(lambda change: self.handle_event(change, 'm2_dy'), 'value')
        self.m2_Rx_control.observe(lambda change: self.handle_event(change, 'm2_Rx'), 'value')
        self.m2_Ry_control.observe(lambda change: self.handle_event(change, 'm2_Ry'), 'value')
        self.cam_dz_control.observe(lambda change: self.handle_event(change, 'cam_dz'), 'value')
        self.cam_dx_control.observe(lambda change: self.handle_event(change, 'cam_dx'), 'value')
        self.cam_dy_control.observe(lambda change: self.handle_event(change, 'cam_dy'), 'value')
        self.cam_Rx_control.observe(lambda change: self.handle_event(change, 'cam_Rx'), 'value')
        self.cam_Ry_control.observe(lambda change: self.handle_event(change, 'cam_Ry'), 'value')
        for i in range(20):
            self.m1m3_controls[i].observe(lambda change, i=i: self.handle_event(change, ('m1m3_vals', i)), 'value')
            self.m2_controls[i].observe(lambda change, i=i: self.handle_event(change, ('m2_vals', i)), 'value')
        self.zero_control.on_click(self.zero)
        self.randomize_control.on_click(self.randomize)
        self.reveal_control.on_click(self.reveal)
        self.solve_control.on_click(self.solve)
        self.control_truncated_control.on_click(self.control_truncated)
        self.control_penalty_control.on_click(self.control_penalty)

        self.view = self._view()
        self.textout = ipywidgets.Textarea(
            value=self.text,
            layout=ipywidgets.Layout(height="250pt", width="auto")
        )
        self._pause_handler = False
        self._is_playing = False
        self._control_history = []

    def zero(self, b):
        self.m2_dz = 0.0
        self.m2_dx = 0.0
        self.m2_dy = 0.0
        self.m2_Rx = 0.0
        self.m2_Ry = 0.0
        self.cam_dz = 0.0
        self.cam_dx = 0.0
        self.cam_dy = 0.0
        self.cam_Rx = 0.0
        self.cam_Ry = 0.0
        self.m1m3_vals = [0.0]*20
        self.m2_vals = [0.0]*20
        self.offsets = np.zeros_like(self.randomized_dof)
        self.text = 'Values Zeroed!'
        self._is_playing = False
        self._n_iter = 0
        self.update()
        self._control_history = []
        self._control_history.append(
            (
                self.wfe,
                self.m2_dz,
                self.m2_dx,
                self.m2_dy,
                self.m2_Rx,
                self.m2_Ry,
                self.cam_dz,
                self.cam_dx,
                self.cam_dy,
                self.cam_Rx,
                self.cam_Ry,
                self.m1m3_vals,
                self.m2_vals,
            )
        )

    def randomize(self, b):
        # amplitudes for all 50 dof
        amp = [25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0]
        amp += [0.1]*20
        amp += [0.2]*20
        offsets = self.offset_rng.normal(scale=amp)[self.randomized_dof]
        self.offsets = np.round(offsets, 2)

        self.m2_dz = 0.0
        self.m2_dx = 0.0
        self.m2_dy = 0.0
        self.m2_Rx = 0.0
        self.m2_Ry = 0.0
        self.cam_dz = 0.0
        self.cam_dx = 0.0
        self.cam_dy = 0.0
        self.cam_Rx = 0.0
        self.cam_Ry = 0.0
        self.m1m3_vals = [0.0]*20
        self.m2_vals = [0.0]*20
        self.text = 'Values Randomized!'
        self._is_playing = True
        self._n_iter = 0
        self.update()
        self._control_history = []
        self._control_history.append(
            (
                self.wfe,
                self.m2_dz,
                self.m2_dx,
                self.m2_dy,
                self.m2_Rx,
                self.m2_Ry,
                self.cam_dz,
                self.cam_dx,
                self.cam_dy,
                self.cam_Rx,
                self.cam_Ry,
                self.m1m3_vals,
                self.m2_vals,
            )
        )

    def reveal(self, b):
        self.text = ""
        self.text += f"M2 dz: {self.offsets[0]:.2f} µm\n\n"
        self.text += f"M2 dx: {self.offsets[1]:.2f} µm\n\n"
        self.text += f"M2 dy: {self.offsets[2]:.2f} µm\n\n"
        self.text += f"M2 Rx: {self.offsets[3]:.2f} arcsec\n\n"
        self.text += f"M2 Ry: {self.offsets[4]:.2f} arcsec\n\n"
        self.text += f"Cam dz: {self.offsets[5]:.2f} µm\n\n"
        self.text += f"Cam dx: {self.offsets[6]:.2f} µm\n\n"
        self.text += f"Cam dy: {self.offsets[7]:.2f} µm\n\n"
        self.text += f"Cam Rx: {self.offsets[8]:.2f} arcsec\n\n"
        self.text += f"Cam Ry: {self.offsets[9]:.2f} arcsec\n\n"
        self._is_playing = False
        self.update()

    def solve(self, b):
        self._is_playing = False
        self.m2_dz = 0.0
        self.m2_dx = 0.0
        self.m2_dy = 0.0
        self.m2_Rx = 0.0
        self.m2_Ry = 0.0
        self.cam_dz = 0.0
        self.cam_dx = 0.0
        self.cam_dy = 0.0
        self.cam_Rx = 0.0
        self.cam_Ry = 0.0
        self.m1m3_vals = [0.0]*20
        self.m2_vals = [0.0]*20

        for idof, offset in zip(self.controlled_dof, self.offsets):
            match idof:
                case 0:
                    self.m2_dz = -offset
                case 1:
                    self.m2_dx = -offset
                case 2:
                    self.m2_dy = -offset
                case 3:
                    self.m2_Rx = -offset
                case 4:
                    self.m2_Ry = -offset
                case 5:
                    self.cam_dz = -offset
                case 6:
                    self.cam_dx = -offset
                case 7:
                    self.cam_dy = -offset
                case 8:
                    self.cam_Rx = -offset
                case 9:
                    self.cam_Ry = -offset
                case idof if idof in range(10, 30):
                    self.m1m3_vals[idof-10] = -offset
                case idof if idof in range(30, 50):
                    self.m2_vals[idof-30] = -offset
        self.reveal(None)
        self.update()

    def control_truncated(self, b):
        dz_fit = self.fit_dz()
        sens = np.array(self.sens)

        # Don't use M2 tilt or camera piston.
        sens = sens[:, self.controlled_truncated_dof]
        dof_fit, _, _, _ = lstsq(sens, dz_fit)
        dof_fit = np.round(dof_fit, 2)
        full_dof = np.zeros(50)
        full_dof[self.controlled_truncated_dof] = dof_fit
        self.apply_dof(-full_dof)
        self._plot_control_history()

    def control_penalty(self, b):
        amp = [25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0]
        amp += [0.1]*20
        amp += [0.2]*20
        # Add rows to sens matrix to penalize large dof
        dz_fit = self.fit_dz()
        ndz = len(dz_fit)
        sens = np.zeros((ndz+len(self.controlled_dof), len(self.controlled_dof)))
        sens[:ndz, :] = self.sens[:, self.controlled_dof]
        alpha = self.alpha # strength of penalty
        for i in range(ndz, ndz+len(self.controlled_dof)):
            sens[i, i-ndz] = alpha/amp[self.controlled_dof[i-ndz]]
        dof_fit, _, _, _ = lstsq(sens, np.concatenate([dz_fit, [0]*len(self.controlled_dof)]))
        dof_fit = np.round(dof_fit, 2)
        full_dof = np.zeros(50)
        full_dof[self.controlled_dof] = dof_fit
        with self.debug:
            print(f"{dz_fit.shape = }")
            print(f"{sens.shape = }")
            print(f"{dof_fit.shape = }")
            print(f"{full_dof.shape = }")
            print(f"WFE resids = {sens[:ndz] @ dof_fit - dz_fit}")
            print(f"penalty = {sens[ndz:] @ dof_fit}")

        self.apply_dof(-full_dof)
        self._plot_control_history()

    def _plot_control_history(self):
        self._control_history.append(
            (
                self.wfe,
                self.m2_dz,
                self.m2_dx,
                self.m2_dy,
                self.m2_Rx,
                self.m2_Ry,
                self.cam_dz,
                self.cam_dx,
                self.cam_dy,
                self.cam_Rx,
                self.cam_Ry,
                self.m1m3_vals,
                self.m2_vals,
            )
        )
        with self.control_log:
            with contextlib.suppress(AttributeError):
                fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 2))
                axes[0].plot([x[0] for x in self._control_history], c='k')
                axes[1].plot([x[1] for x in self._control_history], c='b')
                axes[1].plot([x[6] for x in self._control_history], c='r')
                axes[2].plot([x[2] for x in self._control_history], c='b', ls='--')
                axes[2].plot([x[3] for x in self._control_history], c='b', ls=':')
                axes[2].plot([x[7] for x in self._control_history], c='r', ls='--')
                axes[2].plot([x[8] for x in self._control_history], c='r', ls=':')
                axes[3].plot([x[4] for x in self._control_history], c='b', ls='--')
                axes[3].plot([x[5] for x in self._control_history], c='b', ls=':')
                axes[3].plot([x[9] for x in self._control_history], c='r', ls='--')
                axes[3].plot([x[10] for x in self._control_history], c='r', ls=':')

                axes[0].set_ylabel("WFE (µm)")
                axes[1].set_ylabel("dz (µm)")
                axes[2].set_ylabel("dx, dy (µm)")
                axes[3].set_ylabel("Rx, Ry (arcsec)")
                for ax in axes:
                    ax.set_xlabel("Iteration")

                fig.tight_layout()
                plt.show(fig)

    def fit_dz(self):
        # Wavefront estimation part of the control loop.
        Rubin_mask_params = yaml.safe_load(open(Path(danish.datadir)/'ComCamObsc.yaml'))
        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_mask_params,
            focal_length=10.31, pixel_scale=10e-6
        )
        sky_level = 1.0

        # dz_terms = (
        #     (1, 4),                          # defocus
        #     (2, 4), (3, 4),                  # field tilt
        #     (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
        #     (1, 7), (1, 8),                  # constant coma
        #     # (1, 9), (1, 10),                 # constant trefoil
        #     # (1, 11),                         # constant spherical
        #     # (1, 12), (1, 13),                # second astigmatism
        #     # (1, 14), (1, 15),                # quatrefoil
        #     # (1, 16), (1, 17),
        #     # (1, 18), (1, 19),
        #     # (1, 20), (1, 21),
        #     # (1, 22)
        # )
        dz_terms = self.dz_terms

        thxs = []
        thys = []
        z_refs = []
        imgs = []
        names = []
        for sensor in self._sensors.values():
            if sensor.focusz == 0.0:
                continue
            thxs.append(np.deg2rad(sensor.thx))
            thys.append(np.deg2rad(sensor.thy))
            z_refs.append(sensor.z_ref)
            imgs.append(sensor.img.get_array().data)
            names.append(sensor.name)

        fitter = danish.MultiDonutModel(
            factory, z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(0.4), thxs=thxs, thys=thys
        )
        nstar = len(thxs)
        guess = [0.0]*nstar + [0.0]*nstar + [0.5] + [0.0]*len(dz_terms)
        sky_levels = [sky_level]*nstar

        with self.control_log:
            print()
            result = least_squares(
                fitter.chi, guess, jac=fitter.jac,
                ftol=1e-3, xtol=1e-3, gtol=1e-3,
                max_nfev=20, verbose=2,
                args=(imgs, sky_levels)
            )

        dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)

        with self.control_log:
            with contextlib.suppress(AttributeError):
                mods = fitter.model(
                    dxs_fit, dys_fit, fwhm_fit, dz_fit
                )
                fig, axes = plt.subplots(nrows=4, ncols=9, figsize=(9, 4))
                for i in range(18):
                    j = 2 * (i//9)
                    axes[j,i%9].imshow(imgs[i]/np.sum(imgs[i]))
                    axes[j+1,i%9].imshow(mods[i]/np.sum(mods[i]))
                    axes[j,i%9].set_title(names[i])
                for ax in axes.ravel():
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal')
                axes[0, 0].set_ylabel('Data')
                axes[1, 0].set_ylabel('Model')
                axes[2, 0].set_ylabel('Data')
                axes[3, 0].set_ylabel('Model')
                fig.tight_layout()
                plt.show(fig)

            print()
            print("Found double Zernikes:")
            for z_term, val in zip(dz_terms, dz_fit):
                print(f"DZ({z_term[0]}, {z_term[1]}): {val*1e9:10.2f} nm")

        return dz_fit

    @cached_property
    def sens(self):
        amp = [25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0]
        amp += [0.1]*20
        amp += [0.2]*20
        dz_terms = self.dz_terms
        ndz = len(dz_terms)
        sens = np.zeros((ndz, 50))
        jmax = max([j for k, j in dz_terms])
        kmax = max([k for k, j in dz_terms])
        dz_ref = batoid.doubleZernike(
            self.fiducial, np.deg2rad(0.4), self.wavelength,
            jmax=jmax, kmax=kmax, eps=0.61
        )*self.wavelength
        for idof in range(50):
            dof = np.zeros(50)
            dof[idof] = amp[idof]  # microns or arcsec
            builder = self.builder.with_aos_dof(dof.tolist())
            telescope = builder.build()
            dz_p = batoid.doubleZernike(
                telescope, np.deg2rad(0.4), self.wavelength,
                jmax=jmax, kmax=kmax, eps=0.61
            )*self.wavelength
            for i, (k, j) in enumerate(dz_terms):
                sens[i, idof] = (dz_p - dz_ref)[k, j]/amp[idof]
        return sens

    def apply_dof(self, dof):
        with self.control_log:
            print()
            print("Applying DOF:")
            print(f"M2 dz:  {dof[0]:10.2f} µm")
            print(f"M2 dx:  {dof[1]:10.2f} µm")
            print(f"M2 dy:  {dof[2]:10.2f} µm")
            print(f"M2 Rx:  {dof[3]:10.2f} arcsec")
            print(f"M2 Ry:  {dof[4]:10.2f} arcsec")
            print(f"cam dz: {dof[5]:10.2f} µm")
            print(f"cam dx: {dof[6]:10.2f} µm")
            print(f"cam dy: {dof[7]:10.2f} µm")
            print(f"cam Rx: {dof[8]:10.2f} arcsec")
            print(f"cam Ry: {dof[9]:10.2f} arcsec")
            for i in range(20):
                print(f"M1M3 mode {i}: {dof[10+i]:10.2f} µm")
            for i in range(20):
                print(f"M2 mode {i}: {dof[30+i]:10.2f} µm")
        self._is_playing = False
        self.m2_dz += dof[0]
        self.m2_dx += dof[1]
        self.m2_dy += dof[2]
        self.m2_Rx += dof[3]
        self.m2_Ry += dof[4]
        self.cam_dz += dof[5]
        self.cam_dx += dof[6]
        self.cam_dy += dof[7]
        self.cam_Rx += dof[8]
        self.cam_Ry += dof[9]
        self.m1m3_vals = [x+y for x, y in zip(self.m1m3_vals, dof[10:30])]
        self.m2_vals = [x+y for x, y in zip(self.m2_vals, dof[30:50])]
        self.update()

    def handle_event(self, change, attr):
        if self._pause_handler:
            return
        if isinstance(attr, tuple):
            current = getattr(self, attr[0])
            current[attr[1]] = change['new']
            setattr(self, attr[0], current)
        else:
            setattr(self, attr, change['new'])
        if self._is_playing:
            self._n_iter += 1
        self.update()

    def _view(self):
        sensspec = [["S00", "S10", "S20"],
                    ["S01", "S11", "S21"],
                    ["S02", "S12", "S22"]]

        with plt.ioff():
            intra_out = ipywidgets.Output()
            focal_out = ipywidgets.Output()
            extra_out = ipywidgets.Output()

            with intra_out:
                self._intra_fig = plt.figure(constrained_layout=True, figsize=(4, 4))
                self._intra_axes = self._intra_fig.subplot_mosaic(sensspec)
                # Determine spacing
                center = (self._intra_axes["S11"].transAxes + self._intra_fig.transFigure.inverted()).transform([0.5, 0.5])
                s01 = (self._intra_axes["S01"].transAxes + self._intra_fig.transFigure.inverted()).transform([0.5, 0.5])
                dx = s01[0] - center[0]  # make this 0.25 degrees
                factor = 0.25/dx
            with focal_out:
                self._focal_fig = plt.figure(constrained_layout=True, figsize=(4, 4))
                self._focal_axes = self._focal_fig.subplot_mosaic(sensspec)
            with extra_out:
                self._extra_fig = plt.figure(constrained_layout=True, figsize=(4, 4))
                self._extra_axes = self._extra_fig.subplot_mosaic(sensspec)

        self._sensors = {}
        for fig, axes, focusz, prefix, ctx in zip(
            [self._intra_fig, self._focal_fig, self._extra_fig],
            [self._intra_axes, self._focal_axes, self._extra_axes],
            [-1.5e-3, 0.0, 1.5e-3],
            ["in", "foc", "ex"],
            [intra_out, focal_out, extra_out]
        ):
            with contextlib.ExitStack() as stack:
                stack.enter_context(ctx)
                for k, ax in axes.items():
                    ax.set_xticks([])
                    ax.set_yticks([])

                    mytrans = ax.transAxes + fig.transFigure.inverted()
                    x, y = mytrans.transform([0.5, 0.5])
                    if focusz == 0.0:
                        nphot = 1000
                        nx = 21
                    else:
                        nphot = 50000
                        nx = 181
                    thx=-(x-center[0])*factor
                    thy=(y-center[1])*factor

                    self._sensors[prefix+k] = Sensor(
                        prefix+k, thx, thy, nphot=nphot, fiducial=self.fiducial,
                        focusz=focusz,
                        img=ax.imshow(np.zeros((nx, nx)), vmin=0, vmax=1)
                    )

                    ax.text(0.02, 0.92, k, transform=ax.transAxes, fontsize=6, color='white')

        # self.wfe_text = fig.text(0.04, 0.89, "WFE", ha="left", va="center", fontsize=16)

        with intra_out:
            self._intra_canvas = self._intra_fig.canvas
            self._intra_canvas.header_visible = False
            self._intra_fig.show()
        with focal_out:
            self._focal_canvas = self._focal_fig.canvas
            self._focal_canvas.header_visible = False
            self._focal_fig.show()
        with extra_out:
            self._extra_canvas = self._extra_fig.canvas
            self._extra_canvas.header_visible = False
            self._extra_fig.show()

        out = ipywidgets.Tab()

        out.children = [intra_out, focal_out, extra_out]
        out.set_title(0, "Intra")
        out.set_title(1, "Focal")
        out.set_title(2, "Extra")

        return out

    def update(self):
        dof = [self.m2_dz, self.m2_dx, self.m2_dy, self.m2_Rx, self.m2_Ry]
        dof += [self.cam_dz, self.cam_dx, self.cam_dy, self.cam_Rx, self.cam_Ry]
        dof += self.m1m3_vals
        dof += self.m2_vals
        dof = np.array(dof)
        dof[self.randomized_dof] += self.offsets
        dof = dof.tolist()

        builder = self.builder.with_aos_dof(dof)
        telescope = builder.build()

        self.dz = batoid.doubleZernike(
            telescope,
            np.deg2rad(0.4),
            500e-9,
            kmax=6,
            jmax=37,
            eps=0.61
        )
        self.wfe = np.sqrt(np.sum(np.square(self.dz[:, 4:])))
        self.wfe *= 500e-9*1e6  # microns

        for sensor in self._sensors.values():
            sensor.draw(telescope, seeing=0.5, rng=self.rng)
        self._intra_canvas.draw_idle()
        self._focal_canvas.draw_idle()
        self._extra_canvas.draw_idle()
        # self.wfe_text.set_text(f"WFE = {self.wfe:.3f} µm   iter: {self._n_iter}")
        # if self._is_playing:
        #     if self.wfe < 0.5:
                # self.win_text.set_text("You Win!")
                # self._is_playing = False

        self.textout.value = self.text

        self._pause_handler = True
        self.m2_dz_control.value = self.m2_dz
        self.m2_dx_control.value = self.m2_dx
        self.m2_dy_control.value = self.m2_dy
        self.m2_Rx_control.value = self.m2_Rx
        self.m2_Ry_control.value = self.m2_Ry
        self.cam_dz_control.value = self.cam_dz
        self.cam_dx_control.value = self.cam_dx
        self.cam_dy_control.value = self.cam_dy
        self.cam_Rx_control.value = self.cam_Rx
        self.cam_Ry_control.value = self.cam_Ry
        for i, val in enumerate(self.m1m3_vals):
            self.m1m3_controls[i].value = val
        for i, val in enumerate(self.m2_vals):
            self.m2_controls[i].value = val
        self._pause_handler = False

    def display(self):
        from IPython.display import display
        self.app = ipywidgets.HBox([
            self.view,
            self.controls,
            self.textout
        ])

        self.update()
        return self.app
