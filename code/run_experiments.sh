#!/bin/bash

# Constants for fixed hyperparameters
SEED=17
THREADS=1
BACKBONE="bert-base-uncased"
LEARNING_RATE=5e-05
DROPOUT=0.1
WEIGHT_DECAY=0.01
SAVE_WEIGHTS=False

# Hyperparameters to tune
BATCH_SIZES=(16 32)
EPOCHS=(2 3 4)
LR_SCHEDULES=("linear" "cosine" "None")  # Using "None" as a string
LABEL_SMOOTHINGS=(0.0 0.1)

# Loop through all combinations of the hyperparameters
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
        for LR_SCHEDULE in "${LR_SCHEDULES[@]}"; do
            for LABEL_SMOOTH in "${LABEL_SMOOTHINGS[@]}"; do
                echo "Running with batch_size=$BATCH_SIZE, epochs=$EPOCH, lr_schedule=$LR_SCHEDULE, label_smoothing=$LABEL_SMOOTH"
                python3 sentiment_analysis.py \
                    --batch_size "$BATCH_SIZE" \
                    --epochs "$EPOCH" \
                    --seed "$SEED" \
                    --threads "$THREADS" \
                    --backbone "$BACKBONE" \
                    --learning_rate "$LEARNING_RATE" \
                    --lr_schedule "$LR_SCHEDULE" \
                    --dropout "$DROPOUT" \
                    --weight_decay "$WEIGHT_DECAY" \
                    --label_smoothing "$LABEL_SMOOTH" \
                    --save_weights "$SAVE_WEIGHTS"
            done
        done
    done
done
