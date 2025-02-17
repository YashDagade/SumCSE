#!/bin/bash
#SBATCH --job-name=SumCSE_ORTHO_2_07
#SBATCH --output=./results_new/ortho/02_07_ORTHO.log
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --ntasks=4
#SBATCH --mem=130gb
#SBATCH --time=08:00:00

# Load Anaconda and activate the sumcse environment
module load anaconda
source /home/users/yd211/anaconda3/etc/profile.d/conda.sh
conda activate sumcse

# Change to the working directory where your code resides
cd /usr/project/xtmp/yd211/Documents/SumCSE

# Run the training script with ortho loss parameters.
./scripts/simcse_train_test.sh 4 ../result/SumCSE_ortho_2_07/ \
  --model_name_or_path roberta-large \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 128 \
  --train_file ../Data/SumCSE.csv \
  --num_train_epochs 3 \
  --ortho_loss_percent 1.0 \
  --ortho_margin 0.0
