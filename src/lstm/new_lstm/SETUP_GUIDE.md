# LSTM Model Setup - Complete Guide

## 📁 What Has Been Created

I've created a complete LSTM-based multiple-choice question answering system in the `src/lstm/new_lstm/` directory. Here's what was added:

### Files Created:
1. **`lstm_model.py`** - LSTM model architecture
2. **`train_lstm.py`** - Training script
3. **`predict_lstm.py`** - Inference script
4. **`run_training.sh`** - Quick start training script (executable)
5. **`run_inference.sh`** - Quick start inference script (executable)
6. **`README.md`** - Detailed documentation
7. **`__init__.py`** - Package initialization
8. **`SETUP_GUIDE.md`** - This guide

## 🎯 Key Differences from DeBERTa

| Feature | LSTM (New Model) | DeBERTa (Existing) |
|---------|------------------|-------------------|
| **Pretrained Weights** | ❌ None - trained from scratch | ✅ Yes - uses pretrained weights |
| **Architecture** | LSTM + Attention | Transformer |
| **Training Time** | Longer (starts from random) | Shorter (fine-tuning) |
| **Model Size** | Smaller (~10M params) | Larger (~300M+ params) |
| **Memory Usage** | Lower | Higher |
| **Dependencies** | Uses your `preprocessing.py` & `embeddings.py` | Uses HuggingFace transformers |

## 🏗️ Architecture Overview

```
Input: [batch, 4 choices, seq_len]
    ↓
Token Embeddings (from src/embeddings.py)
    ↓
Positional Encoding
    ↓
Dropout
    ↓
Bidirectional LSTM (2 layers)
    ↓
Attention Mechanism (weighted pooling)
    ↓
Classification Head
    ↓
Output: [batch, 4] logits for each choice
```

## 🚀 Quick Start

### Option 1: Using Shell Scripts (Recommended)

```bash
# Navigate to project root
cd "/Users/kshitijpatil/Desktop/AML Project/AppliedML_Group8"

# Train the model
./src/lstm/new_lstm/run_training.sh

# Generate predictions
./src/lstm/new_lstm/run_inference.sh
```

### Option 2: Using Python Directly

**Training:**
```bash
cd "/Users/kshitijpatil/Desktop/AML Project/AppliedML_Group8"

python src/lstm/new_lstm/train_lstm.py \
    --train_path data/train.csv \
    --tokenizer_name roberta-base \
    --output_dir src/lstm/new_lstm/checkpoints \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 1e-3
```

**Inference:**
```bash
python src/lstm/new_lstm/predict_lstm.py \
    --test_path data/test.csv \
    --checkpoint_path src/lstm/new_lstm/checkpoints/best_model.pt \
    --output_path src/lstm/new_lstm/submission.csv
```

## 📊 Expected Training Output

```
✅ Using Mac GPU (MPS)
📂 Loading training data from data/train.csv ...
🔀 Performing stratified train/validation split ...
🔧 Loading tokenizer: roberta-base
   (Note: Using tokenizer vocab only, NO pretrained weights)
🧪 Tokenizing train and validation datasets ...
✅ Train samples: 3712, Val samples: 928

🏗️  Initializing LSTM model from scratch...
   - Vocab size: 50265
   - Embedding dim: 256
   - Hidden size: 256
   - Num layers: 2
   - Bidirectional: True
   - Total parameters: 17,853,444
   - Trainable parameters: 17,853,444

🚀 Starting training for 10 epochs...
================================================================================

Epoch 1/10
Batch [50/464] Loss: 1.3624 Accuracy: 32.81%
...
📈 Training - Loss: 1.2156, Accuracy: 42.15%
📊 Validation - Loss: 1.1234, Accuracy: 45.67%
💾 Saved best model to checkpoints/best_model.pt
```

## 🔧 Customization

### Adjust Model Size

```bash
# Larger model (more capacity)
python src/lstm/new_lstm/train_lstm.py \
    --d_model 512 \
    --hidden_size 512 \
    --num_layers 3

# Smaller model (faster, less memory)
python src/lstm/new_lstm/train_lstm.py \
    --d_model 128 \
    --hidden_size 128 \
    --num_layers 1
```

### Change Training Parameters

```bash
# Longer training with smaller batches
python src/lstm/new_lstm/train_lstm.py \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 5e-4

# More aggressive training
python src/lstm/new_lstm/train_lstm.py \
    --num_epochs 15 \
    --batch_size 16 \
    --learning_rate 2e-3
```

## 📈 Monitoring Training

The training script will output:
- Loss and accuracy every 50 batches
- Validation metrics after each epoch
- Best model saves automatically
- All checkpoints saved for each epoch

## 💡 How It Uses Existing Code

1. **`src/preprocessing.py`**: 
   - Uses `load_data()` to load CSV files
   - Uses `split_dataset()` for stratified train/val split
   - Uses `preprocess_mc_batch()` for tokenization

2. **`src/embeddings.py`**:
   - Uses `TokenEmbedding` class for trainable embeddings
   - Uses `PositionalEncoding` class for position information

3. **`src/config.py`**:
   - Uses `EMBED_D_MODEL`, `EMBED_MAX_LEN`, `EMBED_DROPOUT`
   - Uses `EMBED_INIT` for initialization strategy

## 🎓 Training from Scratch vs Fine-tuning

**Training from Scratch (This LSTM):**
- ✅ No dependency on pretrained models
- ✅ Smaller model size
- ✅ Full control over architecture
- ✅ Learns task-specific representations
- ❌ Requires more training data
- ❌ Takes longer to converge
- ❌ May have lower accuracy initially

**Fine-tuning (DeBERTa):**
- ✅ Faster convergence
- ✅ Better initial performance
- ✅ Leverages pretrained knowledge
- ❌ Larger model
- ❌ Higher memory requirements
- ❌ Black-box pretrained weights

## 🐛 Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in the correct directory
cd "/Users/kshitijpatil/Desktop/AML Project/AppliedML_Group8"

# Install required packages
pip install torch transformers datasets pandas scikit-learn
```

### Issue: Out of memory
```bash
# Reduce batch size
python src/lstm/new_lstm/train_lstm.py --batch_size 4

# Or reduce model size
python src/lstm/new_lstm/train_lstm.py --hidden_size 128
```

### Issue: Model not learning
```bash
# Try lower learning rate
python src/lstm/new_lstm/train_lstm.py --learning_rate 5e-4

# Or train longer
python src/lstm/new_lstm/train_lstm.py --num_epochs 20
```

## 📂 Directory Structure After Setup

```
src/lstm/new_lstm/
├── __init__.py
├── lstm_model.py          # Model architecture
├── train_lstm.py          # Training script
├── predict_lstm.py        # Inference script
├── run_training.sh        # Quick training script
├── run_inference.sh       # Quick inference script
├── README.md              # Detailed documentation
├── SETUP_GUIDE.md         # This guide
└── checkpoints/           # Created during training
    ├── best_model.pt      # Best model by validation accuracy
    ├── checkpoint_epoch_1.pt
    ├── checkpoint_epoch_2.pt
    └── ...
```

## 🎯 Next Steps

1. **Start Training**: Run `./src/lstm/new_lstm/run_training.sh`
2. **Monitor Progress**: Watch the training metrics
3. **Generate Predictions**: Run `./src/lstm/new_lstm/run_inference.sh`
4. **Compare Results**: Compare with DeBERTa baseline
5. **Tune Hyperparameters**: Experiment with different settings
6. **Ensemble**: Consider combining with other models

## 📚 Additional Resources

- See `README.md` for detailed API documentation
- Check `lstm_model.py` for architecture details
- Refer to `train_lstm.py` for training loop implementation
- Look at `predict_lstm.py` for inference logic

## ✅ Verification Checklist

- [x] Model architecture defined
- [x] Training script created
- [x] Inference script created
- [x] Uses existing preprocessing
- [x] Uses existing embeddings
- [x] No pretrained weights used
- [x] Device agnostic (CPU/GPU/MPS)
- [x] Proper error handling
- [x] Documentation complete
- [x] Quick start scripts ready

## 🎉 You're All Set!

The LSTM model is ready to train. It will:
- Train all weights from scratch (no pretrained models)
- Use your existing preprocessing and embedding code
- Save checkpoints automatically
- Generate predictions on test data

Start with: `./src/lstm/new_lstm/run_training.sh`
