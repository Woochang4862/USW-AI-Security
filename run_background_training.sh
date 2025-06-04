#!/bin/bash

# Background Training Script for Interpretable MMTD
# Usage: ./run_background_training.sh [classifier_type] [checkpoint_path]

CLASSIFIER_TYPE=${1:-"logistic_regression"}
CHECKPOINT_PATH=${2:-"MMTD/checkpoints/fold1/checkpoint-939/pytorch_model.bin"}
OUTPUT_DIR="outputs/background_training_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting background training..."
echo "📊 Classifier: $CLASSIFIER_TYPE"
echo "💾 Checkpoint: $CHECKPOINT_PATH"
echo "📁 Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training in background with nohup
nohup python scripts/background_training.py \
    --classifier_type "$CLASSIFIER_TYPE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --max_samples 1000 \
    > "$OUTPUT_DIR/nohup.out" 2>&1 &

PID=$!
echo "🔄 Training started with PID: $PID"
echo "📝 Logs: $OUTPUT_DIR/nohup.out"
echo "📊 Progress: $OUTPUT_DIR/logs/progress_*.log"

# Save PID for monitoring
echo $PID > "$OUTPUT_DIR/training.pid"

echo ""
echo "📋 Monitoring commands:"
echo "  tail -f $OUTPUT_DIR/nohup.out                    # Main output"
echo "  tail -f $OUTPUT_DIR/logs/progress_*.log          # Progress log"
echo "  kill $PID                                        # Stop training"
echo "  ps aux | grep $PID                               # Check status"

echo ""
echo "✅ Background training started successfully!"
