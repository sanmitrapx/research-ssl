#!/bin/bash
#SBATCH --job-name=scale_law
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scale_%a.out
#SBATCH --error=logs/scale_%a.out
#SBATCH --array=0-5

cd /home/sanmitra/research-ssl

SIZES=(5 10 20 50 100 200)
N=${SIZES[$SLURM_ARRAY_TASK_ID]}

echo "=== Scaling law: N_train=${N}, 100 epochs, CosineAnnealingLR ==="

python scripts/finetune_jakubnet.py \
    --no-deltas \
    --bins 128 \
    --grid 0.01 \
    --train-split "data/jakubnet_splits/jakubnet_scale_${N}.txt" \
    --test-split "data/jakubnet_splits/jakubnet_ft_test_benchmark.txt" \
    --output-json "outputs/scaling_law/scale_${N}.json"

echo "=== Done: N_train=${N} ==="
