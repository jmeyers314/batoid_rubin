# Improve the bending mode representation.  Don't remove the relative PTT between
# M1 and M3.  Use annular Zernikes.  Use separate Zernikes for M1 and M3.

[ ! -d improved ] && mkdir improved

python M1M3_NASTRAN_to_norm.py --M1ptt 6 --M3ptt 0 improved/M1M3_norm_improved.pkl
python M1M3_norm_to_sag.py --input improved/M1M3_norm_improved.pkl improved/M1M3_sag_improved.pkl
python M1M3_decompose_sag.py --input improved/M1M3_sag_improved.pkl improved/M1M3_decomposition_improved.pkl
python M1M3_format_for_batoid.py --input improved/M1M3_decomposition_improved.pkl --outdir improved --nkeep 30


python M2_NASTRAN_to_norm.py --M2ptt 6 improved/M2_norm_improved.pkl
python M2_norm_to_sag.py --input improved/M2_norm_improved.pkl improved/M2_sag_improved.pkl
python M2_decompose_sag.py --input improved/M2_sag_improved.pkl improved/M2_decomposition_improved.pkl
python M2_format_for_batoid.py --input improved/M2_decomposition_improved.pkl --outdir improved --nkeep 30
