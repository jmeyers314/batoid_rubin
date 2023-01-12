import os

import batoid
import batoid_rubin
import numpy as np


fea_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "batoid_rubin",
    "data",
    "fea"
)

bend_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "batoid_rubin",
    "data",
    "bend_legacy"
)


def test_fea_nodes_load():
    bx, by, idx1, idx3 = batoid_rubin.builder.m1m3_fea_nodes(fea_dir)
    bx, by = batoid_rubin.builder.m2_fea_nodes(fea_dir)


def test_grid_xy_load():
    m1_grid_xy, m3_grid_xy = batoid_rubin.builder.m1m3_grid_xy(bend_dir)
    m2_grid_xy = batoid_rubin.builder.m2_grid_xy(bend_dir)


def test_fea():
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    grav = batoid_rubin.builder.m1m3_gravity(fea_dir, telescope, 0.1)
    temp = batoid_rubin.builder.m1m3_temperature(fea_dir, 0.0, 0.0, 0.0, 0.0, 0.0)
    lut = batoid_rubin.builder.m1m3_lut(fea_dir, 0.1, 0.0, 0)

    grav = batoid_rubin.builder.m2_gravity(fea_dir, 0.1)
    temp = batoid_rubin.builder.m2_temperature(fea_dir, 0.0, 0.0)


def test_load_bend():
    dof = (0,)*20
    m1_bend = batoid_rubin.builder.realize_bend(bend_dir, dof, 0)
    m2_bend = batoid_rubin.builder.realize_bend(bend_dir, dof, 1)
    m3_bend = batoid_rubin.builder.realize_bend(bend_dir, dof, 2)


def test_builder():
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    builder = batoid_rubin.builder.LSSTBuilder(fiducial, fea_dir, bend_dir)
    builder = (
        builder
        .with_m1m3_gravity(0.1)
        .with_m1m3_temperature(0.0, 0.1, -0.1, 0.1, 0.1)
        .with_m2_gravity(0.1)
        .with_m2_temperature(0.1, 0.1)
        .with_aos_dof(np.array([0]*19+[1]+[0]*30))
        .with_m1m3_lut(0.1, 0.0, 0)
    )

    telescope = builder.build()
