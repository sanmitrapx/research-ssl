#!/bin/bash
set -e

echo "=== Launching ablation pipeline ==="

# Stage 1: Color feature ablation
JOB1=$(sbatch --parsable scripts/ablate_color.sh)
echo "Stage 1 (Color): Job ${JOB1}"

# Stage 2: Loss x Bins ablation (depends on Stage 1)
JOB2=$(sbatch --parsable --dependency=afterany:${JOB1} scripts/ablate_loss_bins.sh)
echo "Stage 2 (Loss x Bins): Job ${JOB2} (depends on ${JOB1})"

# Stage 3: Grid resolution ablation (depends on Stage 2)
JOB3=$(sbatch --parsable --dependency=afterany:${JOB2} scripts/ablate_grid.sh)
echo "Stage 3 (Grid): Job ${JOB3} (depends on ${JOB2})"

echo ""
echo "=== Pipeline submitted ==="
echo "Stage 1 (Color):      ${JOB1}"
echo "Stage 2 (Loss x Bins): ${JOB2}  (after ${JOB1})"
echo "Stage 3 (Grid):        ${JOB3}  (after ${JOB2})"
echo ""
echo "Monitor with: squeue -u $(whoami)"
