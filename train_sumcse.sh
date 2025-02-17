#!/bin/bash
#SBATCH --job-name=SumCSE_Train       # Job name
#SBATCH --output=%x_%j.out            # Output file, %x=job-name, %j=job-ID
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=compsci-gpu       # Partition (queue) to use
#SBATCH --gres=gpu:a5000:1            # Request one A5000 GPU
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --mem=40G                     # Memory allocation
#SBATCH --time=02:00:00               # Time limit (adjust as needed)

# Load environment and activate conda
module load anaconda                # Load Anaconda module (if required by your system)
source activate sumcse              # Activate your conda environment

# Change to your working directory
cd ~/Documents/SumCSE

# Run your training script
python train.py \
    --model_name_or_path roberta-large \
    --train_file ../Data/SumCSE.csv \
    --output_dir ../result/SumCSE_ortho-l0.5_m0.05 \
    --per_device_train_batch_size 64 \
    --num_train_epochs 3 \
    --ortho_loss_lambda 0.5 \
    --ortho_margin 0.05 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --do_train \
    --do_eval
