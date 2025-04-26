#!/bin/bash
#SBATCH -J cmr
#SBATCH -p bme_a10080g
#SBATCH -e /home_data/home/liyuyang/data/output/%x_%j.e
#SBATCH -o /home_data/home/liyuyang/data/output/%x_%j.o
#SBATCH -N 1
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mem=80G


# env
source ~/.bashrc
conda activate cmr25
# module load cuda/7/12.2
export WANDB_MODE="offline"

# task
python main.py fit \
    --config configs/base.yaml \
    --config configs/model/pmr-plus.yaml \
    --config configs/train/pmr-plus/cmr25-cardiac.yaml 