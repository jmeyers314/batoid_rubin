# Generate bending modes by matching previously determined bending mode forces to
# the unit load cases.  Subtract PTT individually from M1, M2, and M3.

NASTRAN_DIR=/Users/josh/src/ZEMAX_FEMAP
OUT_DIR=bend_ptt

[ ! -d $OUT_DIR ] && mkdir $OUT_DIR

python M1M3_match_forces.py --M1ptt 3 --M3ptt 3 --indir $NASTRAN_DIR $OUT_DIR/M1M3_bend_ptt.asdf
python M1M3_decompose_sag.py --input $OUT_DIR/M1M3_bend_ptt.asdf $OUT_DIR/M1M3_decomposition_bend_ptt.asdf
python M1M3_format_for_batoid.py --input $OUT_DIR/M1M3_decomposition_bend_ptt.asdf --outdir $OUT_DIR --nkeep 30 --do_forces


python M2_match_forces.py --M2ptt 3 --indir $NASTRAN_DIR $OUT_DIR/M2_bend_ptt.asdf
python M2_decompose_sag.py --input $OUT_DIR/M2_bend_ptt.asdf $OUT_DIR/M2_decomposition_bend_ptt.asdf
python M2_format_for_batoid.py --input $OUT_DIR/M2_decomposition_bend_ptt.asdf --outdir $OUT_DIR --nkeep 30 --do_forces
