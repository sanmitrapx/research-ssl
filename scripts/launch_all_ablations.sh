#!/bin/bash
set -e

echo "=== Launching ablation pipeline (priorities 1-3) ==="
echo ""

# All three axes run in parallel -- no dependencies needed
JOB1=$(sbatch --parsable scripts/ablate_bins.sh)
echo "Bins ablation:  Job ${JOB1}  (128, 256 bins)"

JOB2=$(sbatch --parsable scripts/ablate_loss.sh)
echo "Loss ablation:  Job ${JOB2}  (pure CE, ordinal_kl)"

JOB3=$(sbatch --parsable scripts/ablate_grid.sh)
echo "Grid ablation:  Job ${JOB3}  (grid=0.01)"

echo ""
echo "=== 5 runs submitted (all independent) ==="
echo "  Bins:  ${JOB1}  [0]=128bins  [1]=256bins"
echo "  Loss:  ${JOB2}  [0]=pure_ce  [1]=ordinal_kl"
echo "  Grid:  ${JOB3}  grid=0.01"
echo ""
echo "Already have baselines:"
echo "  physics/64bins/ce_emd/grid=0.02  -> val_rl2=0.242"
echo "  curv_only/64bins/ce_emd/grid=0.02 -> val_rl2=0.243"
echo ""
echo "Monitor with: squeue -u $(whoami)"
