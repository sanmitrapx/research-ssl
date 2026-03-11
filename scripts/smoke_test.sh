#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

echo "Smoke test start: $(date)"

srun python src/train.py \
    data.grid_size=0.02 \
    data.batch_size=4 \
    data.num_cp_bins=64 \
    data.color_mode=physics \
    model.num_cp_bins=64 \
    model.num_concat_levels=4 \
    model.upcast_dim=1232 \
    model.loss_type=ce_emd \
    model.lambda_emd=0.1 \
    model.lambda_recon=0.0 \
    trainer.max_epochs=1 \
    trainer.devices=1 \
    trainer.strategy=auto \
    experiment_name=smoke_test

echo "Smoke test end: $(date)"
echo "Exit code: $?"
