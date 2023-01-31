import contextlib

import batoid
import batoid_rubin
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np


class Raft:
    def __init__(self, name, thx, thy, nphot, img, wavelength=500e-9):
        self.name = name
        self.thx = thx
        self.thy = thy
        self.nphot = nphot
        self.img = img
        self.bins = img.get_array().shape[0]
        bo2 = self.bins//2
        self.range = [[-bo2*10e-6, bo2*10e-6], [-bo2*10e-6, bo2*10e-6]]
        self.wavelength = wavelength

    def draw(self, telescope, seeing):
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
            nrandom=self.nphot,
            theta_x=np.deg2rad(self.thx), theta_y=np.deg2rad(self.thy)
        )
        telescope.trace(rv)

        rv.x[:] -= np.mean(rv.x[~rv.vignetted])
        rv.y[:] -= np.mean(rv.y[~rv.vignetted])

        # Convolve in a Gaussian
        scale = 10e-6 * seeing/2.35/0.2
        rv.x[:] += np.random.normal(scale=scale, size=len(rv))
        rv.y[:] += np.random.normal(scale=scale, size=len(rv))

        # Bin rays
        psf, _, _ = np.histogram2d(
            rv.y[~rv.vignetted], rv.x[~rv.vignetted], bins=self.bins,
            range=self.range
        )
        self.img.set_array(psf/np.max(psf))


class AlignGame:
    def __init__(self, debug=None):
        if debug is None:
            debug = contextlib.redirect_stdout(None)
        self.debug = debug

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

        self.controls = ipywidgets.VBox([
            self.m2_dz_control, self.m2_dx_control, self.m2_dy_control,
            self.m2_Rx_control, self.m2_Ry_control,
            self.cam_dz_control, self.cam_dx_control, self.cam_dy_control,
            self.cam_Rx_control, self.cam_Ry_control,
            self.zero_control, self.randomize_control,
            self.reveal_control, self.solve_control
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

        self.view = self._view()
        self.textout = ipywidgets.Textarea(
            value=self.text,
            layout=ipywidgets.Layout(height="250pt", width="auto")
        )
        self._pause_handler = False

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
        self.update()

    def randomize(self, b):
        # amplitudes
        amp = [25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0]
        offsets = np.random.normal(scale=amp)
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
        self.update()

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
        self.update()

    def solve(self, b):
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

    def handle_event(self, change, attr):
        if self._pause_handler:
            return
        setattr(self, attr, change['new'])
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
                k, thx, thy, nphot=nphot,
                img=ax.imshow(np.zeros((nx, nx)), vmin=0, vmax=1)
            )

        self.wfe_text = fig.text(0.31, 0.89, "WFE", ha="left", va="center", fontsize=16)

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
            raft.draw(telescope, seeing=0.5)
        self.wfe_text.set_text(f"WFE = {self.wfe:.3f} µm")
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
