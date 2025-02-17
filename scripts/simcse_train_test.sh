#!/bin/bash
#! IMP: first argument should be the number of GPUs. Second should be the output directory.
#! This version uses torchrun (which passes LOCAL_RANK via the environment) instead of torch.distributed.launch.
#! That avoids injecting unwanted '--local-rank' arguments into train.py’s argument parser.

# Capture positional parameters:
NUM_GPU=$1
OUTPUT_DIR=$2
shift 2

# (Optional) You can filter out any residual local-rank flags—but torchrun should not pass them.
CLEAN_ARGS=()
for arg in "$@"; do
  if [[ $arg != --local_rank* ]]; then
    CLEAN_ARGS+=("$arg")
  fi
done

# Set a random master port (based on the SLURM job id)
PORT_ID=$(( $SLURM_JOBID % 62536 + 2500 ))
export OMP_NUM_THREADS=1

if [[ ${NUM_GPU} -eq 2 ]]; then
   # Use torchrun for 2 GPUs
   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
     --max_seq_length 32 \
     --evaluation_strategy steps \
     --metric_for_best_model stsb_spearman \
     --load_best_model_at_end \
     --eval_steps 125 \
     --pooler_type cls \
     --overwrite_output_dir \
     --temp 0.05 \
     --do_train \
     --do_eval \
     --fp16 \
     --seed 42 \
     --hard_negative_weight 0 \
     --output_dir "$OUTPUT_DIR" "${CLEAN_ARGS[@]}"
elif [[ ${NUM_GPU} -eq 4 ]]; then
   # Use torchrun for 4 GPUs
   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
     --max_seq_length 32 \
     --evaluation_strategy steps \
     --metric_for_best_model stsb_spearman \
     --load_best_model_at_end \
     --eval_steps 125 \
     --pooler_type cls \
     --overwrite_output_dir \
     --temp 0.05 \
     --do_train \
     --do_eval \
     --fp16 \
     --seed 42 \
     --hard_negative_weight 0 \
     --output_dir "$OUTPUT_DIR" "${CLEAN_ARGS[@]}"
else
   # For a single-GPU run, simply call train.py
   python train.py \
     --max_seq_length 32 \
     --evaluation_strategy steps \
     --metric_for_best_model stsb_spearman \
     --load_best_model_at_end \
     --eval_steps 125 \
     --pooler_type cls \
     --overwrite_output_dir \
     --temp 0.05 \
     --do_train \
     --do_eval \
     --fp16 \
     --seed 42 \
     --hard_negative_weight 0 \
     --output_dir "$OUTPUT_DIR" "${CLEAN_ARGS[@]}"
fi

# Finally, run the evaluation script using the output directory.
./scripts/eval.sh "$OUTPUT_DIR"
