#!/bin/bash -l
#SBATCH --job-name=SumCSE_ORIGINAL_2_08
#SBATCH --output=./results_new/original/02_08_ORIGINAL_A6.log
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=4
#SBATCH --mem=130gb
#SBATCH --time=08:00:00

# (Optional) Load Anaconda and activate the environment.
# Even if "module" is not found, if your environment is set up correctly, this may not block execution.
module load anaconda
source /home/users/yd211/anaconda3/etc/profile.d/conda.sh
conda activate sumcse

# Change to the working directory where your code resides.
cd /usr/project/xtmp/yd211/Documents/SumCSE

# Run the training script for the original experiment.
# The first two positional parameters are: (1) number of GPUs (4) and (2) output directory.
./scripts/simcse_train_test.sh --num_gpus 4 --output_dir ../result/SumCSE_ortho_2_08_A6/ --model_name_or_path roberta-large --learning_rate 1e-5 --per_device_train_batch_size 32 --train_file ../Data/SumCSE.csv --num_train_epochs 3
