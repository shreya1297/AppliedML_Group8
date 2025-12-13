# LSTM Multiple Choice Model (Trained from Scratch)

This folder contains an LSTM-based model for multiple-choice question answering, trained entirely from scratch without using any pretrained weights.

## Architecture

The model consists of:
1. **Token Embeddings**: Trainable embeddings initialized from scratch
2. **Positional Encoding**: Sinusoidal positional encodings
3. **Bidirectional LSTM**: 2-layer bidirectional LSTM
4. **Attention Mechanism**: Attention over LSTM outputs for better context representation
5. **Classification Head**: Final layer to predict the correct answer choice (0-3)

## Key Features

- ✅ **No Pretrained Weights**: All parameters trained from scratch
- ✅ **Custom Embeddings**: Uses the embedding module from `src/embeddings.py`
- ✅ **Preprocessing Integration**: Uses preprocessing from `src/preprocessing.py`
- ✅ **Attention Mechanism**: Weighted pooling over sequence for better representations
- ✅ **Device Agnostic**: Supports CPU, CUDA, and MPS (Mac GPU)

## Files

- `lstm_model.py`: LSTM model architecture
- `train_lstm.py`: Training script
- `predict_lstm.py`: Inference script for generating predictions
- `README.md`: This file

## Installation

Make sure you have the required dependencies:

```bash
pip install torch transformers datasets pandas scikit-learn
```

## Usage

### Training

Train the model from scratch:

```bash
cd /Users/kshitijpatil/Desktop/AML\ Project/AppliedML_Group8

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
    --learning_rate 1e-3 \
    --seed 42 \
    --bidirectional
```

**Important Parameters:**
- `--train_path`: Path to training CSV file
- `--tokenizer_name`: Which tokenizer to use (only for vocabulary, no weights loaded)
- `--output_dir`: Directory to save checkpoints
- `--d_model`: Embedding dimension (default: 256)
- `--hidden_size`: LSTM hidden size (default: 256)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate (default: 1e-3)

### Inference

Generate predictions on test data:

```bash
python src/lstm/new_lstm/predict_lstm.py \
    --test_path data/test.csv \
    --checkpoint_path src/lstm/new_lstm/checkpoints/best_model.pt \
    --tokenizer_name roberta-base \
    --output_path src/lstm/new_lstm/submission.csv \
    --batch_size 16
```

## Model Details

### Input Format
- **Input Shape**: `[batch_size, num_choices=4, seq_len]`
- Each question has 4 answer choices
- Context and question+answer are concatenated and tokenized

### Training Process
1. Load and preprocess data using existing `preprocessing.py`
2. Initialize embeddings from scratch (no pretrained weights)
3. Train with cross-entropy loss
4. Use gradient clipping to prevent exploding gradients
5. Save checkpoints after each epoch
6. Track best model based on validation accuracy

### Output Format
- **Output Shape**: `[batch_size, num_choices=4]`
- Logits for each of the 4 answer choices
- Prediction: argmax of logits (0, 1, 2, or 3)

## Performance Notes

Since this model is trained from scratch (no pretrained embeddings or weights):
- Training will take longer than fine-tuning pretrained models
- May require more epochs to converge
- Performance might be lower than models using pretrained weights (like DeBERTa)
- Benefits: Complete control over architecture and no dependency on external models

## Comparison with DeBERTa

| Aspect | LSTM (This Model) | DeBERTa |
|--------|------------------|---------|
| **Pretrained Weights** | ❌ None (trained from scratch) | ✅ Yes |
| **Embedding Quality** | Lower (random init) | Higher (pretrained) |
| **Training Time** | Faster per epoch | Slower per epoch |
| **Convergence** | Requires more epochs | Faster convergence |
| **Model Size** | Smaller (~5-10M params) | Larger (~300M+ params) |
| **Memory Usage** | Lower | Higher |

## Tips for Better Performance

1. **Increase training epochs**: Since embeddings start random, more epochs help
2. **Tune learning rate**: Try 5e-4, 1e-3, or 2e-3
3. **Experiment with model size**: Increase `hidden_size` or `num_layers`
4. **Data augmentation**: Consider augmenting training data
5. **Ensemble**: Combine with other models for better results

## Troubleshooting

**Issue**: Model not learning (accuracy stuck at ~25%)
- Try lower learning rate (5e-4)
- Increase number of epochs
- Check if gradients are flowing (use gradient clipping)

**Issue**: Out of memory
- Reduce `batch_size`
- Reduce `hidden_size` or `num_layers`
- Reduce `max_length`

**Issue**: Overfitting
- Increase `dropout`
- Use data augmentation
- Reduce model capacity

## Directory Structure After Training

```
src/lstm/new_lstm/
├── lstm_model.py
├── train_lstm.py
├── predict_lstm.py
├── README.md
├── checkpoints/
│   ├── best_model.pt (best validation accuracy)
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   └── ...
└── submission.csv (generated after inference)
```

## Citation

This implementation uses:
- Custom LSTM architecture
- Embeddings from `src/embeddings.py`
- Preprocessing from `src/preprocessing.py`
- Configuration from `src/config.py`
