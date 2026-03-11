#!/bin/bash
#SBATCH --job-name=color_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/color_abl_%A_%a.out
#SBATCH --error=logs/color_abl_%A_%a.err

GRID=0.02
BINS=64
LOSS=ce_emd
EPOCHS=100

case $SLURM_ARRAY_TASK_ID in
    0) COLOR=physics    ; EXP_NAME="color_physics"   ;;
    1) COLOR=curv_only  ; EXP_NAME="color_curv"      ;;
    2) COLOR=flow_only  ; EXP_NAME="color_flow"      ;;
    3) COLOR=lap_only   ; EXP_NAME="color_lap"       ;;
esac

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Experiment: ${EXP_NAME}  color_mode=${COLOR}"
echo "grid=${GRID}, bins=${BINS}, loss=${LOSS}, epochs=${EPOCHS}"
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
