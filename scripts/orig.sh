# Starting from the M1M3_ML files, decompose sag into Zernikes + residual grid.
# This preserves the choice to independently PTT-corrected M1 and M3.
# We do improve things by using annular Zernikes and the correct M2 and M3 radii.
# So this is similar to the "improved" case, but with indep PTT and minus signs
# consistent with the project's original SVD.  We haven't reordered modes here.

[ ! -d orig ] && mkdir orig

python M1M3_decompose_sag.py --input /Users/josh/src/M1M3_ML/data/myUdn3norm_156.mat orig/M1M3_decomposition_orig.pkl
python M1M3_format_for_batoid.py --input orig/M1M3_decomposition_orig.pkl --outdir orig --nkeep 30

python M2_decompose_sag.py --input /Users/josh/src/M2_FEA/data/M2_Udn3norm.mat orig/M2_decomposition_orig.pkl
python M2_format_for_batoid.py --input orig/M2_decomposition_orig.pkl --outdir orig --nkeep 30
