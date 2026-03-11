#!/bin/bash
#SBATCH --job-name=loss_bin
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/loss_bin_abl_%A_%a.out
#SBATCH --error=logs/loss_bin_abl_%A_%a.err

GRID=0.02
COLOR=physics
EPOCHS=100

case $SLURM_ARRAY_TASK_ID in
    0) BINS=32  ; LOSS=ce_emd     ; EXP_NAME="lb_32_ce_emd"       ;;
    1) BINS=64  ; LOSS=ce_emd     ; EXP_NAME="lb_64_ce_emd"       ;;
    2) BINS=128 ; LOSS=ce_emd     ; EXP_NAME="lb_128_ce_emd"      ;;
    3) BINS=64  ; LOSS=ordinal_kl ; EXP_NAME="lb_64_ordinal_kl"   ;;
esac

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Experiment: ${EXP_NAME}  bins=${BINS}  loss=${LOSS}"
echo "grid=${GRID}, color=${COLOR}, epochs=${EPOCHS}"
echo "Node: $(hostname), GPUs: 1"
echo "Start time: $(date)"
echo "========================================="

srun python src/train.py \
    data.grid_size=${GRID} \
    data.batch_size=4 \
    data.num_cp_bins=${BINS} \
    data.color_mode=${COLOR} \
    model.num_cp_bins=${BINS} \
    model.num_concat_levels=4 \
    model.upcast_dim=1232 \
    model.loss_type=${LOSS} \
    model.lambda_emd=0.1 \
    model.lambda_recon=0.0 \
    trainer.max_epochs=${EPOCHS} \
    trainer.devices=1 \
    trainer.strategy=auto \
    experiment_name=${EXP_NAME}

echo "End time: $(date)"
echo "Job completed with exit code: $?"
