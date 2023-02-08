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


class Raft:
    def __init__(self, name, thx, thy, nphot, img, fiducial, wavelength=500e-9):
        self.name = name
        self.thx = thx
        self.thy = thy
        self.nphot = nphot
        self.img = img
        self.fiducial = fiducial
        self.wavelength = wavelength

        self.bins = img.get_array().shape[0]
        bo2 = self.bins//2
        self.range = [[-bo2*10e-6, bo2*10e-6], [-bo2*10e-6, bo2*10e-6]]

        if "in" in self.name:
            telescope = self.fiducial.withGloballyShiftedOptic(
                "Detector", [0, 0, -1.5e-3]
            )
        elif "ex" in self.name:
            telescope = self.fiducial.withGloballyShiftedOptic(
                "Detector", [0, 0, 1.5e-3]
            )
        else:
            telescope = self.fiducial
        self.z_ref = batoid.zernikeTA(
            telescope, np.deg2rad(self.thx), np.deg2rad(self.thy),
            self.wavelength,
            nrad=20, naz=120,
            reference='chief',
            jmax=66, eps=0.61
        )*self.wavelength  # meters

    def draw(self, telescope, seeing, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        if "in" in self.name:
            telescope = telescope.withGloballyShiftedOptic(
                "Detector", [0, 0, -1.5e-3]
            )
        elif "ex" in self.name:
            telescope = telescope.withGloballyShiftedOptic(
                "Detector", [0, 0, 1.5e-3]
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


class AlignGame:
    def __init__(self, debug=None, rng=None, nthread=8):
        batoid._batoid.set_nthreads(nthread)

        if debug is None:
            debug = contextlib.redirect_stdout(None)
        self.debug = debug
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng
        self.offset_rng = np.random.default_rng(rng.integers(2**63))

        self.fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
        self.builder = batoid_rubin.LSSTBuilder(
            self.fiducial,
            fea_dir="fea_legacy",
            bend_dir="bend_legacy"
        )
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

        self.offsets = np.zeros(50)
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

        self.controls = ipywidgets.VBox([
            self.m2_dz_control, self.m2_dx_control, self.m2_dy_control,
            self.m2_Rx_control, self.m2_Ry_control,
            self.cam_dz_control, self.cam_dx_control, self.cam_dy_control,
            self.cam_Rx_control, self.cam_Ry_control,
            self.zero_control, self.randomize_control,
            self.reveal_control, self.solve_control,
            self.control_truncated_control, self.control_penalty_control
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
        self.offsets = np.zeros(50)
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
            )
        )

    def randomize(self, b):
        # amplitudes
        amp = [25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0]
        offsets = self.offset_rng.normal(scale=amp)*2
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
        self.offsets = np.round(np.concatenate([offsets, np.zeros(40)]), 2)
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
        self.m2_dz = -self.offsets[0]
        self.m2_dx = -self.offsets[1]
        self.m2_dy = -self.offsets[2]
        self.m2_Rx = -self.offsets[3]
        self.m2_Ry = -self.offsets[4]
        self.cam_dz = -self.offsets[5]
        self.cam_dx = -self.offsets[6]
        self.cam_dy = -self.offsets[7]
        self.cam_Rx = -self.offsets[8]
        self.cam_Ry = -self.offsets[9]
        self.reveal(None)
        self.update()

    def control_truncated(self, b):
        # Don't use M2 tilt or camera piston.
        dz_fit = self.fit_dz()
        sens = np.array(self.sens)
        sens = sens[:, [0,1,2,6,7,8,9]]
        dof_fit, _, _, _ = lstsq(sens, dz_fit)
        dof_fit = np.round(dof_fit, 2)
        full_dof = np.zeros(10)
        full_dof[[0,1,2,6,7,8,9]] = dof_fit
        self.apply_dof(-full_dof)
        self._plot_control_history()

    def control_penalty(self, b):
        # Add rows to sens matrix to penalize large dof
        dz_fit = self.fit_dz()
        sens = np.zeros((9+10, 10))
        sens[:9, :] = self.sens
        alpha = 1e-12 # strength of penalty
        for i in range(9, 19):
            sens[i, i-9] = alpha
        dof_fit, _, _, _ = lstsq(sens, np.concatenate([dz_fit, [0]*10]))
        dof_fit = np.round(dof_fit, 2)
        self.apply_dof(-dof_fit)
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
            )
        )
        with self.debug:
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
        Rubin_obsc = yaml.safe_load(open(Path(danish.datadir)/'RubinObsc.yaml'))
        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            obsc_radii=Rubin_obsc['radii'], obsc_motion=Rubin_obsc['motion'],
            focal_length=10.31, pixel_scale=10e-6
        )
        sky_level = 1.0

        dz_terms = (
            (1, 4),                          # defocus
            (2, 4), (3, 4),                  # field tilt
            (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
            (1, 7), (1, 8),                  # constant coma
            # (1, 9), (1, 10),                 # constant trefoil
            # (1, 11),                         # constant spherical
            # (1, 12), (1, 13),                # second astigmatism
            # (1, 14), (1, 15),                # quatrefoil
            # (1, 16), (1, 17),
            # (1, 18), (1, 19),
            # (1, 20), (1, 21),
            # (1, 22)
        )

        thxs = []
        thys = []
        z_refs = []
        imgs = []
        names = []
        for raft in self._rafts.values():
            if raft.name.startswith('R'):
                continue
            thxs.append(np.deg2rad(raft.thx))
            thys.append(np.deg2rad(raft.thy))
            z_refs.append(raft.z_ref)
            imgs.append(raft.img.get_array().data[::-1, ::-1])
            names.append(raft.name)

        fitter = danish.MultiDonutModel(
            factory, z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys
        )
        nstar = len(thxs)
        guess = [0.0]*nstar + [0.0]*nstar + [0.5] + [0.0]*len(dz_terms)
        sky_levels = [sky_level]*nstar

        with self.debug:
            print()
            result = least_squares(
                fitter.chi, guess, jac=fitter.jac,
                ftol=1e-3, xtol=1e-3, gtol=1e-3,
                max_nfev=20, verbose=2,
                args=(imgs, sky_levels)
            )

        dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)

        with self.debug:
            with contextlib.suppress(AttributeError):
                mods = fitter.model(
                    dxs_fit, dys_fit, fwhm_fit, dz_fit
                )
                fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(8, 2))
                for i in range(8):
                    axes[0,i].imshow(imgs[i]/np.sum(imgs[i]))
                    axes[1,i].imshow(mods[i]/np.sum(mods[i]))
                    axes[0,i].set_title(names[i])
                for ax in axes.ravel():
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal')
                axes[0, 0].set_ylabel('Data')
                axes[1, 0].set_ylabel('Model')
                fig.tight_layout()
                plt.show(fig)

            print()
            print("Found double Zernikes:")
            for z_term, val in zip(dz_terms, dz_fit):
                print(f"DZ({z_term[0]}, {z_term[1]}): {val*1e9:10.2f} nm")

        return dz_fit

    @cached_property
    def sens(self):
        dz_terms = (
            (1, 4),                          # defocus
            (2, 4), (3, 4),                  # field tilt
            (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
            (1, 7), (1, 8),                  # constant coma
        )
        sens = np.zeros((9, 10))
        dz_ref = batoid.doubleZernike(
            self.fiducial, np.deg2rad(1.8), self.wavelength,
            jmax=9, kmax=4, eps=0.61
        )*self.wavelength
        for idof in range(10):
            dof = np.zeros(50)
            dof[idof] = 100.0  # microns or arcsec
            builder = self.builder.with_aos_dof(dof.tolist())
            telescope = builder.build()
            dz_p = batoid.doubleZernike(
                telescope, np.deg2rad(1.8), self.wavelength,
                jmax=9, kmax=4, eps=0.61
            )*self.wavelength
            for i, (j, k) in enumerate(dz_terms):
                sens[i, idof] = (dz_p - dz_ref)[j, k]/100
        return sens

    def apply_dof(self, dof):
        with self.debug:
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

        self.update()

    def handle_event(self, change, attr):
        if self._pause_handler:
            return
        setattr(self, attr, change['new'])
        if self._is_playing:
            self._n_iter += 1
        self.update()

    def _view(self):
        self._fig = fig = plt.figure(constrained_layout=True, figsize=(5, 5))
        raftspec = [[  None, "in04",  None,  None,  None,   None,   None],
                    [  None, "ex04", "R14", "R24", "R34", "ex44", "in44"],
                    [  None,  "R03", "R13", "R23", "R33",  "R43",   None],
                    [  None,  "R02", "R12", "R22", "R32",  "R42",   None],
                    [  None,  "R01", "R11", "R21", "R31",  "R41",   None],
                    ["in00", "ex00", "R00", "R10", "R20", "ex40",   None],
                    [  None,   None,  None,  None,  None, "in40",   None]]
        self._axes = fig.subplot_mosaic(
            raftspec, empty_sentinel=None
        )

        # Determine spacing
        center = (self._axes["R22"].transAxes + fig.transFigure.inverted()).transform([0.5, 0.5])
        r02 = (self._axes["R02"].transAxes + fig.transFigure.inverted()).transform([0.5, 0.5])
        dx = r02[0] - center[0]  # make this 1.4 degrees
        factor = 1.4/dx

        self._rafts = {}
        for k, ax in self._axes.items():
            ax.set_xticks([])
            ax.set_yticks([])

            mytrans = ax.transAxes + fig.transFigure.inverted()
            x, y = mytrans.transform([0.5, 0.5])
            if "R" in k:
                nphot = 1000
                nx = 21
                thx=(x-center[0])*factor
                thy=(y-center[1])*factor
            else:
                nphot = 50000
                nx = 181
            # Redo WF centering
            if k == "in00":
                thx, thy = -5.25*3.5/15, -5*3.5/15
            elif k == "ex00":
                thx, thy = -4.75*3.5/15, -5*3.5/15

            elif k == "in04":
                thx, thy = -5*3.5/15, 5.25*3.5/15
            elif k == "ex04":
                thx, thy = -5*3.5/15, 4.75*3.5/15

            elif k == "in40":
                thx, thy = 5*3.5/15, -5.25*3.5/15
            elif k == "ex40":
                thx, thy = 5*3.5/15, -4.75*3.5/15

            elif k == "in44":
                thx, thy = 5.25*3.5/15, 5*3.5/15
            elif k == "ex44":
                thx, thy = 4.75*3.5/15, 5*3.5/15

            self._rafts[k] = Raft(
                k, thx, thy, nphot=nphot, fiducial=self.fiducial,
                img=ax.imshow(np.zeros((nx, nx)), vmin=0, vmax=1)
            )

        self.wfe_text = fig.text(0.31, 0.89, "WFE", ha="left", va="center", fontsize=16)
        self.win_text = fig.text(0.31, 0.96, "", ha="left", va="center", fontsize=16, color='red')

        self._canvas = fig.canvas
        self._canvas.header_visible = False

        out = ipywidgets.Output()
        with out:
            plt.show(fig)
        return out

    def update(self):
        dof = [self.m2_dz, self.m2_dx, self.m2_dy, self.m2_Rx, self.m2_Ry]
        dof += [self.cam_dz, self.cam_dx, self.cam_dy, self.cam_Rx, self.cam_Ry]
        dof += [0]*40
        dof = (self.offsets + dof).tolist()

        builder = self.builder.with_aos_dof(dof)
        telescope = builder.build()

        self.dz = batoid.doubleZernike(
            telescope,
            np.deg2rad(1.75),
            500e-9,
            kmax=15,
            jmax=37,
            eps=0.61
        )
        self.wfe = np.sqrt(np.sum(np.square(self.dz[:, 4:])))
        self.wfe *= 500e-9*1e6  # microns

        for raft in self._rafts.values():
            raft.draw(telescope, seeing=0.5, rng=self.rng)
        self.wfe_text.set_text(f"WFE = {self.wfe:.3f} µm   iter: {self._n_iter}")
        if self._is_playing:
            if self.wfe < 0.5:
                self.win_text.set_text("You Win!")
                self._is_playing = False
        self._canvas.draw()

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
        self._pause_handler = False

    def display(self):
        from IPython.display import display
        self.app = ipywidgets.HBox([
            self.view,
            self.controls,
            self.textout

        ])

        display(self.app)
        self.update()
