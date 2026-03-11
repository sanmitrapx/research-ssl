#!/bin/bash
#SBATCH --job-name=bins_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bins_abl_%j.out
#SBATCH --error=logs/bins_abl_%j.err

source /opt/miniconda/etc/profile.d/conda.sh
conda activate sonata

GRID=0.01
COLOR=physics
BINS=128
LOSS=ce_emd
EPOCHS=150
EXP_NAME="bins_128_grid001"

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Experiment: ${EXP_NAME}  bins=${BINS}  grid=${GRID}"
echo "loss=${LOSS}, color=${COLOR}, epochs=${EPOCHS}"
echo "batch_size=1, accumulate_grad_batches=4 (effective=4)"
echo "Node: $(hostname), GPUs: 2"
echo "Start time: $(date)"
echo "========================================="

srun python src/train.py \
    data.grid_size=${GRID} \
    data.batch_size=1 \
    data.num_cp_bins=${BINS} \
    data.color_mode=${COLOR} \
    model.num_cp_bins=${BINS} \
    model.num_concat_levels=4 \
    model.upcast_dim=1232 \
    model.loss_type=${LOSS} \
    model.lambda_emd=0.1 \
    model.lambda_recon=0.0 \
    trainer.max_epochs=${EPOCHS} \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=2 \
    trainer.strategy=ddp_find_unused_parameters_true \
    experiment_name=${EXP_NAME}

echo "End time: $(date)"
echo "Job completed with exit code: $?"
