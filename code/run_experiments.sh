#!/bin/bash

# Define the list of hyperparameters to vary from 
batch_sizes=(16 32)
learning_rates=(3e-5 2e-5 1e-5)

# Other constant hyperparameters
seed=17
threads=1
backbone="roberta-base"
weight_decay=0.01
label_smoothing=0.1  # Fixed to match usage
epochs=10


# Iterate over each combination of hyperparameters
for batch_size in "${batch_sizes[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    # Run the Python script with the current combination
    python3 sentiment_analysis.py \
      --batch_size $batch_size \
      --epochs $epochs \
      --learning_rate $learning_rate \
      --label_smoothing $label_smoothing \
      --seed $seed \
      --threads $threads \
      --backbone $backbone \
      --weight_decay $weight_decay
  done
done