#!/bin/bash
#SBATCH --job-name=reg_rl2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/reg_rl2_%j.out
#SBATCH --error=logs/reg_rl2_%j.err

source /opt/miniconda/etc/profile.d/conda.sh
conda activate sonata

EPOCHS=200

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Experiment: regression_rl2"
echo "  model=sonata_cp_regression  loss=rL2  epochs=${EPOCHS}"
echo "  scheduler=CosineAnnealingLR  layerwise_lr_decay"
echo "  batch_size=1  accumulate_grad_batches=4 (effective=4)"
echo "Node: $(hostname), GPUs: 2"
echo "Start time: $(date)"
echo "========================================="

srun python src/train.py \
    model=sonata_cp_regression \
    data.grid_size=0.02 \
    data.batch_size=1 \
    data.color_mode=physics \
    model.num_concat_levels=4 \
    model.upcast_dim=1232 \
    model.max_epochs=${EPOCHS} \
    trainer.max_epochs=${EPOCHS} \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=2 \
    trainer.strategy=ddp_find_unused_parameters_true \
    experiment_name=regression_rl2

echo "End time: $(date)"
echo "Job completed with exit code: $?"
