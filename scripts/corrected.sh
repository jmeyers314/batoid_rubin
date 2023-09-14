# Reproduce as closely as possible the original bending modes,
# but fix the M2 and M3 radii.  Should precisely reproduce M1.

NASTRAN_DIR=/Users/josh/src/ZEMAX_FEMAP
OUT_DIR=corrected

[ ! -d $OUT_DIR ] && mkdir $OUT_DIR

python M1M3_NASTRAN.py --M1ptt 3 --M3ptt 3 --indir $NASTRAN_DIR $OUT_DIR/M1M3_corrected.pkl
python M1M3_decompose_sag.py --input $OUT_DIR/M1M3_corrected.pkl --zk_simultaneous --circular $OUT_DIR/M1M3_decomposition_corrected.pkl
python M1M3_format_for_batoid.py --input $OUT_DIR/M1M3_decomposition_corrected.pkl --outdir $OUT_DIR --nkeep 30


python M2_NASTRAN.py --M2ptt 3 --indir $NASTRAN_DIR $OUT_DIR/M2_corrected.pkl
python M2_decompose_sag.py --input $OUT_DIR/M2_corrected.pkl --circular $OUT_DIR/M2_decomposition_corrected.pkl
python M2_format_for_batoid.py --input $OUT_DIR/M2_decomposition_corrected.pkl --outdir $OUT_DIR --nkeep 30
