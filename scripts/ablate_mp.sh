#!/bin/bash
#SBATCH --job-name=mp_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/mp_abl_%A_%a.out
#SBATCH --error=logs/mp_abl_%A_%a.err

source /opt/miniconda/etc/profile.d/conda.sh
conda activate sonata

COLOR=physics
GRID=0.02
BINS=64
LOSS=ce_emd
LEMD=0.1
EPOCHS=100
STRATEGY=ddp_find_unused_parameters_true

case $SLURM_ARRAY_TASK_ID in
    0) MODEL=sonata_cp_classifier ; EXP_NAME="baseline_no_mp"    ;;
    1) MODEL=sonata_cp_gcn        ; EXP_NAME="mp_gcn_edgeconv"   ;;
    2) MODEL=sonata_cp_pt_head    ; EXP_NAME="mp_pt_attention"   ;;
    3) MODEL=sonata_cp_meshconv   ; EXP_NAME="mp_mesh_conv"      ;;
    4) MODEL=sonata_cp_diffusion  ; EXP_NAME="mp_diffusion"      ;;
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
