#!/bin/bash
#SBATCH --job-name=lora_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/lora_abl_%A_%a.out
#SBATCH --error=logs/lora_abl_%A_%a.err

source /opt/miniconda/etc/profile.d/conda.sh
conda activate sonata

COLOR=physics
GRID=0.01
EPOCHS=150
RANK=8
ALPHA=16

case $SLURM_ARRAY_TASK_ID in
    0) BINS=64  ; LOSS=ce_emd     ; LEMD=0.1 ; EXP_NAME="lora_64_ce_emd"       ;;
    1) BINS=128 ; LOSS=ce_emd     ; LEMD=0.1 ; EXP_NAME="lora_128_ce_emd"      ;;
    2) BINS=64  ; LOSS=ordinal_kl ; LEMD=0.0 ; EXP_NAME="lora_64_ordinal_kl"   ;;
    3) BINS=128 ; LOSS=ordinal_kl ; LEMD=0.0 ; EXP_NAME="lora_128_ordinal_kl"  ;;
esac

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Experiment: ${EXP_NAME}"
echo "  model=sonata_cp_lora  rank=${RANK}  alpha=${ALPHA}"
echo "  bins=${BINS}  loss=${LOSS}  grid=${GRID}"
echo "  scheduler=OneCycleLR  epochs=${EPOCHS}"
echo "  batch_size=1  accumulate_grad_batches=4 (effective=4)"
echo "Node: $(hostname), GPUs: 2"
echo "Start time: $(date)"
echo "========================================="

srun python src/train.py \
    model=sonata_cp_lora \
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
    model.lora_rank=${RANK} \
    model.lora_alpha=${ALPHA} \
    model.max_epochs=${EPOCHS} \
    trainer.max_epochs=${EPOCHS} \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=2 \
    trainer.strategy=ddp \
    experiment_name=${EXP_NAME}

echo "End time: $(date)"
echo "Job completed with exit code: $?"
