## Notes on FEA files


# M1M3 Bending modes

M1M3_1um_156_grid.fits.gz
    - source = IM/data/M1M3/M1M3_1um_156_grid.txt
    - shape = (5256, 159)
    - Each row is one of 5256 FEA nodes.
    - 0th column is M1M3 disambiguator
    - 1st and 2nd columns are FEA node x and y in M1M3 CS
    - Last 156 columns are bending modes; the z-displacement of each node for each mode.

M1M3_1um_156_force.fits.gz
    - source = IM/data/M1M3/M1M3_1um_156_force.txt
    - shape = (156, 159)
    - Each row is one of 156 bending modes.
    - 0th column is actuator ID
    - 1st and 2nd columns are actuator x and y in M1M3 CS
    - Last 156 columns are forces in Newtons for each mode.


# M1M3 print through

M1M3_dxdydz_zenith.fits.gz
    - source = IM/data/M1M3/M1M3_dxdydz_zenith.npy
    - shape = (5256, 3)
    - Each row is one of 5256 FEA nodes.
    - Columns are dx, dy, dz in M1M3 CS.
    - This is the gravitational "print through" when mirror is zenith pointing

M1M3_dxdydz_horizon.fits.gz
    - source = IM/data/M1M3/M1M3_dxdydz_horizon.npy
    - shape = (5256, 3)
    - Each row is one of 5256 FEA nodes.
    - Columns are dx, dy, dz in M1M3 CS.
    - This is the gravitational "print through" when mirror is horizon pointing

M1M3_force_zenith.fits.gz
    - source = IM/data/M1M3/M1M3_force_zenith.npy
    - shape = (256,)
    - Each row is one of 256 actuators.  (So we consider the x and y actuators here too.)
    - Columns are forces in Newtons.
    - These are the mirror support forces when the mirror is zenith pointing.  (Is this after optimization?  Include LUT or not?)

M1M3_force_horizon.fits.gz
    - source = IM/data/M1M3/M1M3_force_horizon.npy
    - shape = (256,)
    - Each row is one of 256 actuators.  (So we consider the x and y actuators here too.)
    - Columns are forces in Newtons.
    - These are the mirror support forces when the mirror is horizon pointing.  (Is this after optimization?  Include LUT or not?)


# M1M3 Thermal

M1M3_thermal_FEA.fits.gz
    - source = IM/data/M1M3/M1M3_thermal_FEA.npy
    - shape = (5244, 7)
    - Each row is one of 5244 FEA nodes.  (Why aren't these the same as above?  I don't know.)
    - Columns are:
        - 0: Unit-Normalized FEA x
        - 1: Unit-Normalized FEA y
        - 2: Bulk temperature dz coefficient
        - 3: x temperature gradient dz coefficient
        - 3: y temperature gradient dz coefficient
        - 3: z temperature gradient dz coefficient
        - 3: r temperature gradient dz coefficient


# M1M3 Miscellany

M1M3_influence_256.fits.gz
    - source = IM/data/M1M3/M1M3_influence_256.npy
    - shape = (5256, 256)
    - Each row is one of 5256 FEA nodes.
    - Each column is one of 256 actuators.
    - Values are dz/dF for each actuator/node.

M1M3_LUT.fits.gz
    - source = IM/data/M1M3/M1M3_LUT.txt
    - shape = (257, 91)
    - First column is index in degrees (0-90 inclusive).  Last 256 columns are forces in Newtons.
    - Each column is LUT for one value of the elevation index.

M1M3_1000N_UL_shape_156.fits.gz
    - source = IM/data/M1M3/M1M3_1000N_UL_shape_156.npy
    - shape = (5256, 156)
    - Rows must be FEA nodes, columns must be bending modes.
    - Not sure what the purpose is of this one.


# M2 Bending modes

M2_1um_grid.fits.gz
    - source = IM/data/M2/M2_1um_grid.DAT
    - shape = (15984, 75)
    - Each row is one of 15984 FEA nodes.
    - 0th column is node index ?
    - 1st and 2nd columns are FEA node x and y in M2 CS
    - Last 72 columns are bending modes; the z-displacement of each node for each mode.

M2_1um_force.fits.gz
    - source = IM/data/M2/M2_1um_force.DAT
    - shape = (72, 75)
    - Each row is one of 72 bending modes.
    - 0th column is actuator ID
    - 1st and 2nd columns are actuator x and y in M2 CS
    - Last 72 columns are forces in Newtons for each mode.

# M2 print through / thermal

M2_GT_FEA.fits.gz
    - source = IM/data/M2/M2_GT_FEA.txt
    - shape = (9084, 6)
    - Each row is one of 9084 FEA nodes.  (Why aren't these the same as above?  I don't know.)
    - Columns are:
        - 0: Unit-Normalized FEA x
        - 1: Unit-Normalized FEA y
        - 2: Zenith print through dz coefficient
        - 3: Horizon print through dz coefficient
        - 4: z temperature gradient dz coefficient
        - 5: r temperature gradient dz coefficient

