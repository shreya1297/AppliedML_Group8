#!/bin/bash

# Quick start script for training LSTM model from scratch
# Run from the project root directory

echo "🚀 Starting LSTM Training from Scratch"
echo "======================================"
echo ""

# Navigate to project root if not already there
cd "$(dirname "$0")/../../.."

echo "📍 Working directory: $(pwd)"
echo ""

# Check if data files exist
if [ ! -f "data/train.csv" ]; then
    echo "❌ Error: data/train.csv not found!"
    echo "Please ensure you're running this from the project root directory."
    exit 1
fi

echo "✅ Training data found"
echo ""

# Create checkpoints directory if it doesn't exist
mkdir -p src/lstm/new_lstm/checkpoints

echo "🏗️  Training LSTM model with default parameters:"
echo "   - Embedding dim: 256"
echo "   - Hidden size: 256"
echo "   - Num layers: 2 (bidirectional)"
echo "   - Epochs: 10"
echo "   - Batch size: 8"
echo "   - Learning rate: 1e-3"
echo ""

# Run training
python src/lstm/new_lstm/train_lstm.py \
    --train_path data/train.csv \
    --tokenizer_name roberta-base \
    --output_dir src/lstm/new_lstm/checkpoints \
    --val_size 0.2 \
    --max_length 128 \
    --d_model 256 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.1 \
    --num_epochs 10 \
    --batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 1e-3 \
    --seed 42 \
    --bidirectional

echo ""
echo "🎉 Training complete!"
echo "📁 Checkpoints saved to: src/lstm/new_lstm/checkpoints/"
