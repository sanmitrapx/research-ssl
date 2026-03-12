#!/bin/bash
#SBATCH --job-name=rh_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/rh_abl_%A_%a.out
#SBATCH --error=logs/rh_abl_%A_%a.err

source /opt/miniconda/etc/profile.d/conda.sh
conda activate sonata

COLOR=physics
GRID=0.02
BINS=64
LOSS=ce_emd
LEMD=0.1
EPOCHS=200
STRATEGY=ddp_find_unused_parameters_true

case $SLURM_ARRAY_TASK_ID in
    0) MODEL=sonata_cp_subbin          ; EXP_NAME="rh_subbin"          ;;
    1) MODEL=sonata_cp_learnable_bins   ; EXP_NAME="rh_learnable_bins"  ;;
    2) MODEL=sonata_cp_multiscale       ; EXP_NAME="rh_multiscale"      ;;
    3) MODEL=sonata_cp_crf              ; EXP_NAME="rh_crf"             ;;
    4) MODEL=sonata_cp_boundary         ; EXP_NAME="rh_boundary"        ;;
    5) MODEL=sonata_cp_cumulative       ; EXP_NAME="rh_cumulative"      ;;
esac

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Experiment: ${EXP_NAME}"
echo "  model=${MODEL}  bins=${BINS}  loss=${LOSS}  grid=${GRID}"
echo "  scheduler=CosineAnnealingLR  epochs=${EPOCHS}"
echo "  batch_size=1  accumulate_grad_batches=4 (effective=4)"
echo "Node: $(hostname), GPUs: 2"
echo "Start time: $(date)"
echo "========================================="

srun python src/train.py \
    model=${MODEL} \
    data.grid_size=${GRID} \
    data.batch_size=1 \
    data.num_cp_bins=${BINS} \
    data.color_mode=${COLOR} \
    model.num_cp_bins=${BINS} \
    model.num_concat_levels=4 \
    model.upcast_dim=1232 \
    model.loss_type=${LOSS} \
    model.lambda_emd=${LEMD} \
    model.lambda_recon=0.0 \
    model.max_epochs=${EPOCHS} \
    trainer.max_epochs=${EPOCHS} \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=2 \
    trainer.strategy=${STRATEGY} \
    experiment_name=${EXP_NAME}

echo "End time: $(date)"
echo "Job completed with exit code: $?"
