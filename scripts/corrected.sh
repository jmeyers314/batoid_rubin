# Reproduce as closely as possible the original bending modes,
# but fix the M2 and M3 radii.  Should precisely reproduce M1.

[ ! -d corrected ] && mkdir corrected

python M1M3_NASTRAN_to_norm.py --M1ptt 3 --M3ptt 3 corrected/M1M3_norm_corrected.pkl
python M1M3_norm_to_sag.py --input corrected/M1M3_norm_corrected.pkl corrected/M1M3_sag_corrected.pkl
python M1M3_decompose_sag.py --input corrected/M1M3_sag_corrected.pkl --zk_simultaneous --circular corrected/M1M3_decomposition_corrected.pkl
python M1M3_format_for_batoid.py --input corrected/M1M3_decomposition_corrected.pkl --outdir corrected --nkeep 30


python M2_NASTRAN_to_norm.py --M2ptt 3 corrected/M2_norm_corrected.pkl
python M2_norm_to_sag.py --input corrected/M2_norm_corrected.pkl corrected/M2_sag_corrected.pkl
python M2_decompose_sag.py --input corrected/M2_sag_corrected.pkl --circular corrected/M2_decomposition_corrected.pkl
python M2_format_for_batoid.py --input corrected/M2_decomposition_corrected.pkl --outdir corrected --nkeep 30
