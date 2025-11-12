from contextlib import ExitStack

import batoid
import galsim
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import FloatText, HBox, Layout, Output, Tab, VBox

import batoid_rubin


class Sensor:
    def __init__(
        self,
        name,
        thx,
        thy,
        nphot,
        focusz,
        wavelength,
        img,
    ):
        self.name = name
        self.thx = thx
        self.thy = thy
        self.nphot = nphot
        self.focusz = focusz
        self.wavelength = wavelength
        self.img = img
        self.bins = self.img.get_array().shape[0]
        bo2 = self.bins // 2
        self.range = bo2 * 10e-6 * np.array([[-1, 1], [-1, 1]])
        self.wf_img = None

    def draw(self, telescope, seeing, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)

        telescope = telescope.withGloballyShiftedOptic("Detector", [0, 0, self.focusz])

        rv = batoid.RayVector.asPolar(
            optic=telescope,
            wavelength=self.wavelength,
            nrandom=self.nphot,
            rng=rng,
            theta_x=np.deg2rad(self.thx),
            theta_y=np.deg2rad(self.thy),
        )
        telescope.trace(rv)

        rv.x[:] -= np.mean(rv.x[~rv.vignetted])
        rv.y[:] -= np.mean(rv.y[~rv.vignetted])

        # Convolve in a Gaussian
        scale = 10e-6 * seeing / 2.35 / 0.2
        rv.x[:] += rng.normal(scale=scale, size=len(rv))
        rv.y[:] += rng.normal(scale=scale, size=len(rv))

        # Bin rays
        psf, _, _ = np.histogram2d(
            rv.y[~rv.vignetted], rv.x[~rv.vignetted], bins=self.bins, range=self.range
        )
        psf = psf[:, ::-1]
        self.img.set_array(psf / np.max(psf))

    def add_wf_img(self, wf_img):
        self.wf_img = wf_img

    def draw_wf(self, telescope):
        if self.wf_img is None:
            return
        telescope = telescope.withGloballyShiftedOptic("Detector", [0, 0, self.focusz])
        wf = batoid.wavefront(
            telescope,
            np.deg2rad(self.thx),
            np.deg2rad(self.thy),
            self.wavelength,
            reference="mean",
            nx=127,
        )
        wfarr = wf.array
        w = ~wfarr.mask

        # Subtract piston/tip/tilt
        coords = wf.coords
        x = coords[..., 0]
        y = coords[..., 1]

        basis = galsim.zernike.zernikeBasis(
            6,
            x[w],
            y[w],
            R_outer=telescope.pupilSize / 2,
        )
        coefs, _, _, _ = np.linalg.lstsq(basis.T, wfarr[w], rcond=None)
        coefs[4:] = 0.0
        ptt = np.dot(coefs, basis)
        wfarr[w] = wfarr[w] - ptt
        wfarr *= self.wavelength * 1e6 / 2.0  # +/-2 microns range
        self.wf_img.set_array(wfarr[:, ::-1])


class LSSTCamInteract:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng

        # self.fiducial = batoid.Optic.fromYaml("LSST_r_align_holes.yaml")
        self.fiducial = batoid.Optic.fromYaml("LSST_r_baffles.yaml")
        self.builder = batoid_rubin.LSSTBuilder(self.fiducial)
        self.wavelength = 625e-9

        # widget variables
        kwargs = {
            "layout": {"width": "175px"},
            "style": {"description_width": "initial"},
            "format": ".3f",
        }
        controls = []
        hc = []
        for name, step, unit in [
            ("m2_dz", 10, "µm"),
            ("m2_dx", 500, "µm"),
            ("m2_dy", 500, "µm"),
            ("m2_Rx", 10, "arcsec"),
            ("m2_Ry", 10, "arcsec"),
            ("cam_dz", 10, "µm"),
            ("cam_dx", 2000, "µm"),
            ("cam_dy", 2000, "µm"),
            ("cam_Rx", 10, "arcsec"),
            ("cam_Ry", 10, "arcsec"),
        ]:
            setattr(self, name, 0.0)
            control = FloatText(
                value=0.0,
                description=name.replace("m2", "M2").replace("cam", "Cam")+f"({unit})",
                step=step,
                **kwargs
            )
            setattr(self, name+"_control", control)
            # Need a default argument for name so lambda captures current value
            control.observe(lambda change, name=name: self.handle_event(change, name), "value")
            hc.append(control)
        controls.append(hc)

        for mirror in ["m1m3", "m2"]:
            setattr(self, mirror+"_vals", [0.0]*20)
            for slc in [(0, 10), (10, 20)]:
                mc = []
                for i in range(*slc):
                    name = mirror+f"_B{i+1}"
                    description = f"{name} (µm)"
                    description = description.replace("m1m3", "M1M3")
                    description = description.replace("m2", "M2")
                    description = description.replace("_", " ")
                    control = FloatText(
                        value=0.0,
                        description=description,
                        step=0.01,
                        **kwargs
                    )
                    setattr(self, name+"_control", control)
                    control.observe(lambda change, i=i, mirror=mirror: self.handle_event(change, (mirror+"_vals", i)), "value")
                    mc.append(control)
                controls.append(mc)

        self.controls = HBox(
            [VBox(control) for control in controls]
        )
        self.view = self._view()
        self._pause_updates = False

    def handle_event(self, change, attr):
        if isinstance(attr, tuple):
            current = getattr(self, attr[0])
            current[attr[1]] = change["new"]
            setattr(self, attr[0], current)
        else:
            setattr(self, attr, change["new"])
        if not self._pause_updates:
            self.update()

    def _view(self):
        cornerspec = [["R00", "R40"], ["R04", "R44"]]
        scispec = [
            [".", "R10", "R20", "R30", "."],
            ["R01", "R11", "R21", "R31", "R41"],
            ["R02", "R12", "R22", "R32", "R42"],
            ["R03", "R13", "R23", "R33", "R43"],
            [".", "R14", "R24", "R34", "."],
        ]

        self._fig_output = {}
        self._fig = {}
        self._axes = {}
        self._sensors = {}
        with plt.ioff():
            for name, spec, focusz in [
                ("intra", cornerspec, -1.5e-3),
                ("extra", cornerspec, 1.5e-3),
                ("focal", scispec, 0.0),
            ]:
                out = Output()
                self._fig_output[name] = out

                with out:
                    fig = plt.figure(constrained_layout=True, figsize=(3.5, 3.5))
                    self._fig[name] = fig
                    fig.suptitle(name, fontsize=8)
                    fig.canvas.header_visible = False
                    axes = fig.subplot_mosaic(spec)
                    self._axes[name] = axes

                    for k, ax in axes.items():
                        row = int(k[2])
                        col = int(k[1])
                        # Raft = 12000 pixels * 0.2 arcsec = 0.666 degrees
                        thx = -(col - 2) * 0.666
                        thy = -(row - 2) * 0.666
                        # Unless were on a WF sensor, then we're a little closer in
                        if abs(row-2) == abs(col-2) == 2:
                            thx *= 5/6
                            thy *= 5/6
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_aspect("equal")
                        if name in ["intra", "extra"]:
                            img = ax.imshow(
                                np.zeros((181, 181)), vmin=0, vmax=1, origin="lower"
                            )
                        else:
                            img = ax.imshow(
                                np.zeros((41, 41)), vmin=0, vmax=1, origin="lower"
                            )
                        ax.text(
                            0.02,
                            0.02,
                            k,
                            fontsize=6,
                            color="white",
                            transform=ax.transAxes,
                        )
                        self._sensors[name + k] = Sensor(
                            name + k,
                            thx=thx,
                            thy=thy,
                            nphot=10_000 if name == "focal" else 100_000,
                            focusz=focusz,
                            wavelength=self.wavelength,
                            img=img,
                        )
                    fig.show()

            # wf is special
            out = Output()
            self._fig_output["wf"] = out
            with out:
                fig = plt.figure(constrained_layout=True, figsize=(3.5, 3.5))
                self._fig["wf"] = fig
                fig.suptitle("Wavefront", fontsize=8)
                fig.canvas.header_visible = False
                axes = fig.subplot_mosaic(scispec)
                self._axes["wf"] = axes
                for k, ax in axes.items():
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect("equal")
                    wf_img = ax.imshow(
                        np.zeros((127, 127)),
                        vmin=-1,
                        vmax=1,
                        origin="lower",
                        cmap="bwr",
                    )
                    self._sensors["focal" + k].add_wf_img(wf_img)
                    ax.plot([0, 1], [0, 1], "k-", lw=0.5)
                fig.show()

        out = HBox([self._fig_output[n] for n in ["intra", "extra", "focal", "wf"]])
        return out

    def update(self):
        dof = [self.m2_dz, self.m2_dx, self.m2_dy, self.m2_Rx, self.m2_Ry]
        dof += [self.cam_dz, self.cam_dx, self.cam_dy, self.cam_Rx, self.cam_Ry]
        dof += self.m1m3_vals
        dof += self.m2_vals
        dof = np.array(dof)

        builder = self.builder.with_aos_dof(dof)
        telescope = builder.build()
        for sensor in self._sensors.values():
            sensor.draw(telescope, seeing=0.1, rng=1)
            sensor.draw_wf(telescope)

    def set_dof(self, dof, flip_x=False, flip_y=False, flip_z=False, flip_M1M3=False, flip_M2=False, use_degrees=False):
        dof = np.array(dof)
        if flip_x:
            dof[1] *= -1
            dof[3] *= -1
            dof[6] *= -1
            dof[8] *= -1
        if flip_y:
            dof[2] *= -1
            dof[4] *= -1
            dof[7] *= -1
            dof[9] *= -1
        if flip_z:
            dof[0] *= -1
            dof[5] *= -1
        if flip_M1M3:
            dof[10:30] *= -1
        if flip_M2:
            dof[30:50] *= -1
        if use_degrees:
            dof[3] *= 3600
            dof[4] *= 3600
            dof[8] *= 3600
            dof[9] *= 3600

        self._pause_updates = True
        try:
            controls = []
            for i, k in enumerate(["dz", "dx", "dy", "Rx", "Ry"]):
                controls.append(getattr(self, "m2_"+k+"_control"))
                controls.append(getattr(self, "cam_"+k+"_control"))
            for i in range(20):
                controls.append(getattr(self, f"m1m3_B{i+1}_control"))
                controls.append(getattr(self, f"m2_B{i+1}_control"))

            with ExitStack() as stack:
                for control in controls:
                    stack.enter_context(control.hold_trait_notifications())

                for i, k in enumerate(["dz", "dx", "dy", "Rx", "Ry"]):
                    control = getattr(self, "m2_"+k+"_control")
                    control.value = dof[i]

                    control = getattr(self, "cam_"+k+"_control")
                    control.value = dof[i+5]
                for i in range(20):
                    control = getattr(self, f"m1m3_B{i+1}_control")
                    control.value = dof[10+i]
                    control = getattr(self, f"m2_B{i+1}_control")
                    control.value = dof[30+i]
        finally:
            self._pause_updates = False
        self.update()

    def display(self):
        self.app = VBox([self.view, self.controls])
        self.update()
        return self.app
