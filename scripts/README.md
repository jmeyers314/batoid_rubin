# Python Scripts

M1M3_NASTRAN_to_norm.py
M2_NASTRAN_to_norm.py
    Load NASTRAN unit actuator load data, optionally adjust with Piston/Tip/Tilt removals, and decompose into bending modes.  Note that the outputs are motions along the normal direction -- these will need to be reprojected into the sag direction for raytracing programs.


M1M3_norm_to_sag.py
M2_norm_to_sag.py
    Reproject bending modes from surface normal displacements to sag displacements.


M1M3_decompose_sag.py
M2_decompose_sag.py
    Decompose sag displacements into Zernike + grid terms.  Options include:
        - Whether to model the Zernike terms for M1 and M3 independently or simultaneously
        - Whether to use circular or annular Zernikes
        - How many Zernikes to use
        - Number of grid points to use


M1M3_format_for_batoid.py
M2_format_for_batoid.py
    Reformat output for batoid consumption.  Options include:
        - How many bending modes to keep
        - Any mode swaps


## Shell Scripts
orig.sh
    Instead of starting form the NASTRAN results, start from the matlab sag results in the M1M3_ML and M2_FEA repositories.  Improve things from the original Zernike+grid decomposition by:
        - Using annular Zernikes
        - Using the correct M2 and M3 radii

corrected.sh
    Starting from NASTRAN, try to reproduce the project's bending modes.  Fix the M2 and M3 radii mistakes.  Note that since there are arbitrary minus signs in the SVD results, the bending mode signs are not necessarily the same as the original project.

improved.sh
    Starting from NASTRAN, make the following differences wrt corrected.sh:
        - Don't remove the relative PTT between M1 and M3
        - Use annular Zernikes
        - Fit Zernikes to M1 and M3 independently
