#!/bin/bash

# Quick inference script for generating predictions
# Run from the project root directory

echo "🔮 Generating Predictions with LSTM Model"
echo "========================================"
echo ""

# Navigate to project root if not already there
cd "$(dirname "$0")/../../.."

echo "📍 Working directory: $(pwd)"
echo ""

# Check if model checkpoint exists
if [ ! -f "src/lstm/new_lstm/checkpoints/best_model.pt" ]; then
    echo "❌ Error: Model checkpoint not found!"
    echo "Please train the model first using run_training.sh"
    exit 1
fi

echo "✅ Model checkpoint found"
echo ""

# Check if test data exists
if [ ! -f "data/test.csv" ]; then
    echo "❌ Error: data/test.csv not found!"
    exit 1
fi

echo "✅ Test data found"
echo ""

# Run inference
python src/lstm/new_lstm/predict_lstm.py \
    --test_path data/test.csv \
    --checkpoint_path src/lstm/new_lstm/checkpoints/best_model.pt \
    --tokenizer_name roberta-base \
    --output_path src/lstm/new_lstm/submission.csv \
    --batch_size 16

echo ""
echo "🎉 Predictions complete!"
echo "📁 Submission file saved to: src/lstm/new_lstm/submission.csv"
