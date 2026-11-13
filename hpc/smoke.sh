#!/bin/bash
# A few minutes on one GPU: every application and the site builder, at toy sizes, into scratch.
#   sbatch hpc/job.sh hpc/smoke.sh
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export HF_HOME=/data/scratch/acw794/home-cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="$KC_SCRATCH/smoke/${SLURM_JOB_ID:-local}"
rm -rf "$OUT"; mkdir -p "$OUT"
python -m applications.sound_matching.fit --manifest /data/scratch/acw794/data/nsynth/selected/manifest.json \
  --note guitar_acoustic_010-040-050 --families steel --smoke --output "$OUT/sound_matching"
python -m applications.sound_matching.evaluate --run "$OUT/sound_matching" --assets "$OUT/docs/supplement/sound_matching" \
  --output "$OUT/docs/supplement/sound_matching/results.json"
python -m applications.semantic.optimise --pair metallic_dull --grid pilot --starts 2 --max-iter 3 \
  --output "$OUT/semantic/metallic_dull"
python web/site.py --docs "$OUT/docs" --semantic "$OUT/semantic/metallic_dull"
echo "smoke OK: $OUT"
