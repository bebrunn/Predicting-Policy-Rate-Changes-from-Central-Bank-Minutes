#!/bin/bash

# Define the list of hyperparameters to vary from Delvin et al. (2019)
batch_sizes=(16 32)
epochs=(2 3 4)
learning_rates=(5e-5 3e-5 2e-5)
lr_schedules=("linear" "cosine")

# Other constant hyperparameters
seed=17
threads=1
backbone="bert-large-uncased"
weight_decay=0.01
label_smoothing=0.1

# Iterate over each combination of hyperparameters
for batch_size in "${batch_sizes[@]}"; do
  for epoch in "${epochs[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for lr_schedule in "${lr_schedules[@]}"; do          
        # Run the Python script with the current combination
        python3 sentiment_analysis.py \
          --batch_size $batch_size \
          --epochs $epoch \
          --learning_rate $learning_rate \
          --lr_schedule $lr_schedule \
          --label_smoothing $label_smoothing \
          --seed $seed \
          --threads $threads \
          --backbone $backbone \
          --weight_decay $weight_decay
      done
    done
  done
done
