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
    builder = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        fea_dir,
        bend_dir,
        dof_coord_system="ZCS",
        flip_m2_bending_modes=True
    )
    builder = (
        builder
        .with_m1m3_gravity(zen)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(zen)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(zen, 0.0, 0)
        .with_extra_zk([0]*4+[1e-9], 0.61)
        .with_camera_gravity(zen, rot)
        .with_camera_temperature(0.2)
    )

    telescope = builder.build()

    # Check that default dirs work
    builder2 = batoid_rubin.LSSTBuilder(
        fiducial,
        dof_coord_system="ZCS",
        flip_m2_bending_modes=True
    )
    builder2 = (
        builder2
        .with_m1m3_gravity(zen)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(zen)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(zen, 0.0, 0)
        .with_extra_zk([0]*4+[1e-9], 0.61)
        .with_camera_gravity(zen, rot)
        .with_camera_temperature(0.2)
    )
    telescope2 = builder2.build()
    assert telescope == telescope2

    # Check float interface too.
    builder3 = batoid_rubin.LSSTBuilder(
        fiducial,
        dof_coord_system="ZCS",
        flip_m2_bending_modes=True
    )
    builder3 = (
        builder3
        .with_m1m3_gravity(zen.rad)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(zen.rad)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(zen.rad, 0.0, 0)
        .with_extra_zk([0]*4+[1e-9], 0.61)
        .with_camera_gravity(zen.rad, rot.rad)
        .with_camera_temperature(0.2)
    )
    telescope3 = builder3.build()
    assert telescope == telescope3


def test_attr():
    builder = batoid_rubin.LSSTBuilder(
        batoid.Optic.fromYaml("LSST_r.yaml"),
        dof_coord_system="ZCS",
        flip_m2_bending_modes=True
    )
    assert hasattr(builder.with_m1m3_gravity, "_req_params")


def test_ep_phase():
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    builder = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        fea_dir,
        bend_dir,
        dof_coord_system="ZCS",
        flip_m2_bending_modes=True
    )
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
        use_m2_modes=list(range(20)),
        dof_coord_system="ZCS",
        flip_m2_bending_modes=True
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
            use_m2_modes=p2,
            dof_coord_system="ZCS",
            flip_m2_bending_modes=True
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
        use_m2_modes=use_m2_modes,
        dof_coord_system="ZCS",
        flip_m2_bending_modes=True
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

        np.testing.assert_allclose(builder1.dof, builder2.dof)
        np.testing.assert_allclose(builder1.dof, builder3.dof)
        np.testing.assert_equal(len(builder1.dof), 10+len(use_m1m3_modes)+len(use_m2_modes))


def test_coord_sys():
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    rays = batoid.RayVector.asPolar(
        optic=fiducial,
        wavelength=622e-9,
        theta_x=0.01,
        theta_y=0.01,
        nrad=10,
        naz=60,
    )
    with np.testing.assert_warns(FutureWarning):
        builder = batoid_rubin.builder.LSSTBuilder(
            fiducial,
            dof_coord_system=None,
            flip_m1m3_bending_modes=False,
            flip_m2_bending_modes=True
        )
    builder_zcs = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_coord_system="ZCS",
        flip_m1m3_bending_modes=False,
        flip_m2_bending_modes=True
    )
    builder_ocs = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_coord_system="OCS",
        flip_m1m3_bending_modes=False,
        flip_m2_bending_modes=True
    )

    m2_dz = 0.1
    m2_dx = 10.0
    m2_dy = -10.0
    m2_rx = 0.01
    m2_ry = -0.01
    cam_dz = 0.2
    cam_dx = -20.0
    cam_dy = 20.0
    cam_rx = -0.02
    cam_ry = 0.02

    scope1 = builder.with_m2_rigid(
        dz=m2_dz,
        dx=m2_dx,
        dy=m2_dy,
        rx=m2_rx*galsim.arcsec,
        ry=m2_ry*galsim.arcsec
    ).with_camera_rigid(
        dz=cam_dz,
        dx=cam_dx,
        dy=cam_dy,
        rx=cam_rx*galsim.arcsec,
        ry=cam_ry*galsim.arcsec
    ).build()  # Default is ZCS

    # Explicit ZCS
    scope2 = builder_zcs.with_m2_rigid(
        dz=m2_dz,
        dx=m2_dx,
        dy=m2_dy,
        rx=m2_rx*galsim.arcsec,
        ry= m2_ry*galsim.arcsec
    ).with_camera_rigid(
        dz=cam_dz,
        dx=cam_dx,
        dy=cam_dy,
        rx=cam_rx*galsim.arcsec,
        ry=cam_ry*galsim.arcsec
    ).build()

    # OCS
    scope3 = builder_ocs.with_m2_rigid(
        dz=m2_dz,
        dx=m2_dx,
        dy=m2_dy,
        rx=m2_rx*galsim.arcsec,
        ry=m2_ry*galsim.arcsec
    ).with_camera_rigid(
        dz=cam_dz,
        dx=cam_dx,
        dy=cam_dy,
        rx=cam_rx*galsim.arcsec,
        ry=cam_ry*galsim.arcsec
    ).build()

    # Manually flip OCS -> ZCS
    scope4 = builder_ocs.with_m2_rigid(
        dz=-m2_dz,
        dx=-m2_dx,
        dy=m2_dy,
        rx=-m2_rx*galsim.arcsec,
        ry=m2_ry*galsim.arcsec
    ).with_camera_rigid(
        dz=-cam_dz,
        dx=-cam_dx,
        dy=cam_dy,
        rx=-cam_rx*galsim.arcsec,
        ry=cam_ry*galsim.arcsec
    ).build()

    trays1 = scope1.trace(rays.copy())
    trays2 = scope2.trace(rays.copy())
    trays3 = scope3.trace(rays.copy())
    trays4 = scope4.trace(rays.copy())

    np.testing.assert_equal(trays1.r, trays2.r)
    np.testing.assert_equal(trays1.v, trays2.v)
    np.testing.assert_equal(trays1.vignetted, trays2.vignetted)
    np.testing.assert_equal(trays1.failed, trays2.failed)

    assert not np.allclose(trays1.x, trays3.x, atol=1e-12, rtol=0)
    assert not np.allclose(trays1.y, trays3.y, atol=1e-12, rtol=0)
    # z's may be close since intersecting a Detector plane
    assert not np.allclose(trays1.v, trays3.v, atol=1e-12, rtol=0)

    np.testing.assert_equal(trays1.r, trays4.r)
    np.testing.assert_equal(trays1.v, trays4.v)
    np.testing.assert_equal(trays1.vignetted, trays4.vignetted)
    np.testing.assert_equal(trays1.failed, trays4.failed)


def test_mirror_flip():
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    rays = batoid.RayVector.asPolar(
        optic=fiducial,
        wavelength=622e-9,
        theta_x=0.01,
        theta_y=0.01,
        nrad=10,
        naz=60,
    )

    rng = np.random.default_rng(57721566)

    with np.testing.assert_warns(FutureWarning):
        builder = batoid_rubin.builder.LSSTBuilder(
            fiducial,
            dof_coord_system="OCS",
        )
    builder_t_t = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_coord_system="OCS",
        flip_m1m3_bending_modes=True,
        flip_m2_bending_modes=True,
    )
    builder_t_f = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_coord_system="OCS",
        flip_m1m3_bending_modes=True,
        flip_m2_bending_modes=False,
    )
    builder_f_t = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_coord_system="OCS",
        flip_m1m3_bending_modes=False,
        flip_m2_bending_modes=True,
    )
    builder_f_f = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_coord_system="OCS",
        flip_m1m3_bending_modes=False,
        flip_m2_bending_modes=False,
    )

    m1m3_dof = rng.uniform(-1e-6, 1e-6, size=20)
    m2_dof = rng.uniform(-1e-6, 1e-6, size=20)

    scope1 = builder.with_m1m3_bend(m1m3_dof).with_m2_bend(m2_dof).build()
    scope2 = builder_f_t.with_m1m3_bend(m1m3_dof).with_m2_bend(m2_dof).build()
    trays1 = scope1.trace(rays.copy())
    trays2 = scope2.trace(rays.copy())
    np.testing.assert_equal(trays1.r, trays2.r)
    np.testing.assert_equal(trays1.v, trays2.v)
    np.testing.assert_equal(trays1.vignetted, trays2.vignetted)
    np.testing.assert_equal(trays1.failed, trays2.failed)

    scope3 = builder_t_f.with_m1m3_bend(m1m3_dof).with_m2_bend(m2_dof).build()
    trays3 = scope3.trace(rays.copy())

    assert not np.allclose(trays1.x, trays3.x, atol=1e-12, rtol=0)
    assert not np.allclose(trays1.y, trays3.y, atol=1e-12, rtol=0)
    # z's may be close since intersecting a Detector plane
    assert not np.allclose(trays1.v, trays3.v, atol=1e-12, rtol=0)
    # Manually flip; should be equivalent to f_t
    scope4 = builder_t_f.with_m1m3_bend(-m1m3_dof).with_m2_bend(-m2_dof).build()
    trays4 = scope4.trace(rays.copy())
    np.testing.assert_equal(trays1.r, trays4.r)
    np.testing.assert_equal(trays1.v, trays4.v)
    np.testing.assert_equal(trays1.vignetted, trays4.vignetted)
    np.testing.assert_equal(trays1.failed, trays4.failed)

    scope5 = builder_t_t.with_m1m3_bend(m1m3_dof).with_m2_bend(m2_dof).build()
    trays5 = scope5.trace(rays.copy())
    assert not np.allclose(trays1.x, trays5.x, atol=1e-12, rtol=0)
    assert not np.allclose(trays1.y, trays5.y, atol=1e-12, rtol=0)
    # z's may be close since intersecting a Detector plane
    assert not np.allclose(trays1.v, trays5.v, atol=1e-12, rtol=0)
    # Manually flip
    scope6 = builder_t_t.with_m1m3_bend(-m1m3_dof).with_m2_bend(m2_dof).build()
    trays6 = scope6.trace(rays.copy())
    np.testing.assert_equal(trays1.r, trays6.r)
    np.testing.assert_equal(trays1.v, trays6.v)
    np.testing.assert_equal(trays1.vignetted, trays6.vignetted)
    np.testing.assert_equal(trays1.failed, trays6.failed)

    scope7 = builder_f_f.with_m1m3_bend(m1m3_dof).with_m2_bend(m2_dof).build()
    trays7 = scope7.trace(rays.copy())
    assert not np.allclose(trays1.x, trays7.x, atol=1e-12, rtol=0)
    assert not np.allclose(trays1.y, trays7.y, atol=1e-12, rtol=0)
    # z's may be close since intersecting a Detector plane
    assert not np.allclose(trays1.v, trays7.v, atol=1e-12, rtol=0)
    # Manually flip
    scope8 = builder_f_f.with_m1m3_bend(m1m3_dof).with_m2_bend(-m2_dof).build()
    trays8 = scope8.trace(rays.copy())
    np.testing.assert_equal(trays1.r, trays8.r)
    np.testing.assert_equal(trays1.v, trays8.v)
    np.testing.assert_equal(trays1.vignetted, trays8.vignetted)
    np.testing.assert_equal(trays1.failed, trays8.failed)


def test_angle_units():
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    rays = batoid.RayVector.asPolar(
        optic=fiducial,
        wavelength=622e-9,
        theta_x=0.01,
        theta_y=0.01,
        nrad=10,
        naz=60,
    )
    rng = np.random.default_rng(5772156649)
    builder1 = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_angle_units="arcsec",
        use_m1m3_modes=[],
        use_m2_modes=[],
    )
    builder2 = batoid_rubin.builder.LSSTBuilder(
        fiducial,
        dof_angle_units="degree",
        use_m1m3_modes=[],
        use_m2_modes=[],
    )
    dof1 = np.array([0, 0, 0, 10, 13, 0, 0, 0, -20, 10])
    dof2 = np.array([0, 0, 0, 10/3600, 13/3600, 0, 0, 0, -20/3600, 10/3600])
    builder1 = builder1.with_aos_dof(dof1)
    builder2 = builder2.with_aos_dof(dof2)
    scope1 = builder1.build()
    scope2 = builder2.build()

    trays1 = scope1.trace(rays.copy())
    trays2 = scope2.trace(rays.copy())
    np.testing.assert_allclose(trays1.r, trays2.r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(trays1.v, trays2.v, rtol=0, atol=1e-12)
    np.testing.assert_equal(trays1.vignetted, trays2.vignetted)
    np.testing.assert_equal(trays1.failed, trays2.failed)


if __name__ == "__main__":
    test_builder()
    test_attr()
    test_ep_phase()
    test_modes_permutation()
    test_subsys_dof()
    test_coord_sys()
    test_mirror_flip()
    test_angle_units()