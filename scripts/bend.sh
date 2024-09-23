# Generate bending modes by matching previously determined bending mode
# forces (from Bo) to the unit load cases.  No PTT subtraction anywhere
# here.

NASTRAN_DIR=${HOME}/src/ZEMAX_FEMAP
OUT_DIR=bend

[ ! -d $OUT_DIR ] && mkdir $OUT_DIR

python M1M3_match_forces.py --M1ptt 0 --M3ptt 0 --indir $NASTRAN_DIR $OUT_DIR/M1M3_bend.asdf
python M1M3_decompose_sag.py --input $OUT_DIR/M1M3_bend.asdf $OUT_DIR/M1M3_decomposition_bend.asdf
python M1M3_format_for_batoid.py --input $OUT_DIR/M1M3_decomposition_bend.asdf --outdir $OUT_DIR --nkeep 30 --do_forces


python M2_match_forces.py --M2ptt 6 --indir $NASTRAN_DIR $OUT_DIR/M2_bend.asdf
python M2_decompose_sag.py --input $OUT_DIR/M2_bend.asdf $OUT_DIR/M2_decomposition_bend.asdf
python M2_format_for_batoid.py --input $OUT_DIR/M2_decomposition_bend.asdf --outdir $OUT_DIR --nkeep 30 --do_forces
