from pathlib import Path

import batoid
import batoid_rubin
import galsim
import numpy as np


fea_dir = Path(batoid_rubin.datadir) / "fea_legacy"
bend_dir = Path(batoid_rubin.datadir) / "bend"

zen = 30 * galsim.degrees
rot = 15 * galsim.degrees


def test_builder():
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    builder = batoid_rubin.builder.LSSTBuilder(fiducial, fea_dir, bend_dir)
    builder = (
        builder
        .with_m1m3_gravity(zen)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(zen)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(zen, 0.0, 0)
        .with_extra_zk([0]*4+[1e-9], 0.61)
    )

    telescope = builder.build()

    # Check that default dirs work
    builder2 = batoid_rubin.LSSTBuilder(fiducial)
    builder2 = (
        builder2
        .with_m1m3_gravity(zen)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(zen)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(zen, 0.0, 0)
        .with_extra_zk([0]*4+[1e-9], 0.61)
    )
    telescope2 = builder2.build()
    assert telescope == telescope2

    # Check float interface too.
    builder3 = batoid_rubin.LSSTBuilder(fiducial)
    builder3 = (
        builder3
        .with_m1m3_gravity(zen.rad)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(zen.rad)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(zen.rad, 0.0, 0)
        .with_extra_zk([0]*4+[1e-9], 0.61)
    )
    telescope3 = builder3.build()
    assert telescope == telescope3


def test_attr():
    builder = batoid_rubin.LSSTBuilder(batoid.Optic.fromYaml("LSST_r.yaml"))
    assert hasattr(builder.with_m1m3_gravity, "_req_params")


def test_ep_phase():
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    builder = batoid_rubin.builder.LSSTBuilder(fiducial, fea_dir, bend_dir)
    builder = (
        builder
        .with_m1m3_gravity(zen)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(zen)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(zen, 0.0, 0)
        .with_extra_zk([0]*4+[1e-9], 0.61)
    )
    telescope = builder.build()
    thx = 0.01
    thy = 0.01
    wavelength=622e-9
    zk = batoid.zernike(
        telescope, thx, thy, wavelength,
        nx=128, jmax=28, eps=0.61
    )
    # Now try to zero-out the wavefront

    builder1 = builder.with_extra_zk(
        zk*wavelength, 0.61
    )
    telescope1 = builder1.build()
    zk1 = batoid.zernike(
        telescope1, thx, thy, wavelength,
        nx=128, jmax=28, eps=0.61
    )

    np.testing.assert_allclose(
        zk1[4:], 0.0, atol=2e-3
    )  # 0.002 waves isn't so bad


def test_modes_permutation():
    """Test that permuting both dof and use_m1m3_modes identically gives same
    result as no permutation.
    """
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    builder1 = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        use_m1m3_modes=list(range(20)),
        use_m2_modes=list(range(20))
    )
    rays = batoid.RayVector.asPolar(
        optic=fiducial,
        wavelength=622e-9,
        theta_x=0.01,
        theta_y=0.01,
        nrad=10,
        naz=60,
    )

    rng = np.random.default_rng(57721)
    for _ in range(10):
        p1 = rng.permutation(20)
        p2 = rng.permutation(20)
        builder2 = batoid_rubin.builder.LSSTBuilder(
            fiducial,
            use_m1m3_modes=p1,
            use_m2_modes=p2
        )
        rigid_dof = np.zeros(10)
        m1m3_dof = rng.uniform(-1e-6, 1e-6, size=20)
        m2_dof = rng.uniform(-1e-6, 1e-6, size=20)
        dof1 = np.concatenate([rigid_dof, m1m3_dof, m2_dof])
        dof2 = np.concatenate([rigid_dof, m1m3_dof[p1], m2_dof[p2]])
        scope1 = builder1.with_aos_dof(dof1).build()
        scope2 = builder2.with_aos_dof(dof2).build()

        trays1 = scope1.trace(rays.copy())
        trays2 = scope2.trace(rays.copy())

        np.testing.assert_allclose(
            trays1.r, trays2.r, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            trays1.v, trays2.v, rtol=0, atol=1e-15
        )
        np.testing.assert_equal(
            trays1.vignetted, trays2.vignetted
        )
        np.testing.assert_equal(
            trays1.failed, trays2.failed
        )

def test_subsys_dof():
    rng = np.random.default_rng(5772156649)
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")

    # Use a subset of the permuted modes to maximally stress the code.
    use_m1m3_modes = rng.permutation(20)[:18]
    use_m2_modes = rng.permutation(20)[:16]
    builder = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        use_m1m3_modes=use_m1m3_modes,
        use_m2_modes=use_m2_modes
    )
    for _ in range(10):
        m2_dz = rng.uniform(-1, 1)
        m2_dx = rng.uniform(-100, 100)
        m2_dy = rng.uniform(-100, 100)
        m2_rx = rng.uniform(-1, 1)
        m2_ry = rng.uniform(-1, 1)

        cam_dz = rng.uniform(-1, 1)
        cam_dx = rng.uniform(-100, 100)
        cam_dy = rng.uniform(-100, 100)
        cam_rx = rng.uniform(-1, 1)
        cam_ry = rng.uniform(-1, 1)

        m1m3_bend = rng.uniform(-0.05, 0.05, size=18)
        m2_bend = rng.uniform(-0.05, 0.05, size=16)

        m2_dof = [m2_dz, m2_dx, m2_dy, m2_rx, m2_ry]
        cam_dof = [cam_dz, cam_dx, cam_dy, cam_rx, cam_ry]
        dof = np.concatenate([
            m2_dof,
            cam_dof,
            m1m3_bend,
            m2_bend
        ])

        builder1 = builder.with_aos_dof(dof)
        builder2 = builder.with_m2_rigid(
            dz=m2_dz, dx=m2_dx, dy=m2_dy,
            rx=m2_rx*galsim.arcsec, ry=m2_ry*galsim.arcsec
        ).with_camera_rigid(
            dz=cam_dz, dx=cam_dx, dy=cam_dy,
            rx=cam_rx*galsim.arcsec, ry=cam_ry*galsim.arcsec
        ).with_m1m3_bend(m1m3_bend).with_m2_bend(m2_bend)
        builder3 = builder.with_m2_rigid(
            dof=m2_dof
        ).with_camera_rigid(
            dof=cam_dof
        ).with_m1m3_bend(m1m3_bend).with_m2_bend(m2_bend)

        scope1 = builder1.build()
        scope2 = builder2.build()
        scope3 = builder3.build()

        assert scope1 == scope2
        assert scope1 == scope3


if __name__ == "__main__":
    test_builder()
    test_attr()
    test_ep_phase()
    test_modes_permutation()
    test_subsys_dof()
