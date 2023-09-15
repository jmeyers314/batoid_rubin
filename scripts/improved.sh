# Improve the bending mode representation.  Don't remove the relative PTT between
# M1 and M3.  Use annular Zernikes.  Use separate Zernikes for M1 and M3.

NASTRAN_DIR=/Users/josh/src/ZEMAX_FEMAP
OUT_DIR=improved

[ ! -d $OUT_DIR ] && mkdir $OUT_DIR

python M1M3_NASTRAN.py --M1ptt 6 --M3ptt 0 --indir $NASTRAN_DIR $OUT_DIR/M1M3_improved.asdf
python M1M3_decompose_sag.py --input $OUT_DIR/M1M3_improved.asdf $OUT_DIR/M1M3_decomposition_improved.asdf
python M1M3_format_for_batoid.py --input $OUT_DIR/M1M3_decomposition_improved.asdf --outdir $OUT_DIR --nkeep 30


python M2_NASTRAN.py --M2ptt 6 --indir $NASTRAN_DIR $OUT_DIR/M2_improved.asdf
python M2_decompose_sag.py --input $OUT_DIR/M2_improved.asdf $OUT_DIR/M2_decomposition_improved.asdf
python M2_format_for_batoid.py --input $OUT_DIR/M2_decomposition_improved.asdf --outdir $OUT_DIR --nkeep 30
