# Starting from the M1M3_ML files, decompose sag into Zernikes + residual grid.
# This preserves the choice to independently PTT-corrected M1 and M3.
# We do improve things by using annular Zernikes and the correct M2 and M3 radii.
# So this is similar to the "improved" case, but with indep PTT and minus signs
# consistent with the project's original SVD.  We haven't reordered modes here.

M1M3_ML_DIR=${HOME}/src/M1M3_ML
M2_FEA_DIR=${HOME}/src/M2_FEA
OUT_DIR=orig

[ ! -d $OUT_DIR ] && mkdir $OUT_DIR

python M1M3_decompose_sag.py --input $M1M3_ML_DIR/data/myUdn3norm_156.mat $OUT_DIR/M1M3_decomposition_orig.asdf
python M1M3_format_for_batoid.py --input $OUT_DIR/M1M3_decomposition_orig.asdf --outdir $OUT_DIR --nkeep 20 --swap "[19,26]"


python M2_decompose_sag.py --input $M2_FEA_DIR/data/M2_Udn3norm.mat $OUT_DIR/M2_decomposition_orig.asdf
python M2_format_for_batoid.py --input $OUT_DIR/M2_decomposition_orig.asdf --outdir $OUT_DIR --nkeep 20 --swap "[17,18,19,25,26,27]"
