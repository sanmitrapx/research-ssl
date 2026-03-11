#!/bin/bash
#SBATCH --job-name=grid_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/grid_abl_%A_%a.out
#SBATCH --error=logs/grid_abl_%A_%a.err

COLOR=physics
BINS=64
LOSS=ce_emd
EPOCHS=100

case $SLURM_ARRAY_TASK_ID in
    0) GRID=0.01 ; EXP_NAME="grid_001"  ;;
    1) GRID=0.04 ; EXP_NAME="grid_004"  ;;
esac

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Experiment: ${EXP_NAME}  grid_size=${GRID}"
echo "bins=${BINS}, loss=${LOSS}, color=${COLOR}, epochs=${EPOCHS}"
echo "Node: $(hostname), GPUs: 1"
echo "Start time: $(date)"
echo "========================================="

srun python src/train.py \
    data.color_mode=${COLOR} \
    data.grid_size=${GRID} \
    data.num_cp_bins=${BINS} \
    model.num_cp_bins=${BINS} \
    model.loss_type=${LOSS} \
    trainer.max_epochs=${EPOCHS} \
    trainer.devices=1 \
    trainer.strategy=auto \
    experiment_name=${EXP_NAME}

echo "End time: $(date)"
echo "Job completed with exit code: $?"
