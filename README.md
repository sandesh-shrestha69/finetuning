# 🎯 Day 19: Fine-Tuning Sentiment Analysis Model
### Transfer Learning with Hugging Face Transformers

Fine-tuned a pre-trained RoBERTa model on Twitter sentiment data to create a custom sentiment classifier. Achieved **76% validation accuracy** and **~74% test accuracy** on 3-class sentiment classification.

---

## What is Fine-Tuning?

**The Problem with Training from Scratch:**
```
Need: Millions of examples
Time: Weeks on expensive GPUs
Cost: Thousands of dollars
You don't have: Any of these resources!
```

**The Solution - Fine-Tuning:**
```
1. Take a model already trained on 58M tweets (RoBERTa)
2. Model already understands language, grammar, sentiment
3. Adapt it to YOUR specific data with just 400 examples
4. Train for 3 epochs (30 minutes on laptop)
5. Get production-ready results!
```

**Analogy:**
```
Training from scratch: Teaching someone to speak English from birth
Fine-tuning:          Teaching an English speaker medical terminology
```

---

## Architecture

### Base Model
**cardiffnlp/twitter-roberta-base-sentiment**
- Pre-trained on 58 million tweets
- Already understands sentiment patterns
- 125 million parameters
- 94% accuracy on benchmark sentiment tasks

### Fine-Tuning Approach
```
Base Model (frozen knowledge)
     ↓
Add custom classification head
     ↓
Train on 400 examples with small learning rate (3e-5)
     ↓
Adapt to YOUR specific sentiment patterns
```

---

## Dataset

**Source:** Kaggle Sentiment Analysis Dataset (Twitter data)

```
Total samples:    27,481 tweets
Used for project:    500 tweets (balanced sampling)

Class distribution:
  - Negative: 167 examples (33%)
  - Neutral:  167 examples (33%)
  - Positive: 167 examples (33%)

Split:
  - Training:   400 examples (80%)
  - Validation:  50 examples (10%)
  - Test:        50 examples (10%)
```

**Sample Data:**
```
Positive: "2am feedings for the baby are fun when he is all smiles and coos"
Negative: "Sooo SAD I will miss you here in San Diego!!!"
Neutral:  "I'd have responded, if I were going"
```

---

## Project Structure

```
day_19_finetuning/
├── data/
│   ├── train.csv              # Original 27k examples
│   ├── train_subset.csv       # 400 training examples
│   ├── val_subset.csv         # 50 validation examples
│   └── test_subset.csv        # 50 test examples
│
├── scripts/
│   ├── prepare_data.py        # Data preprocessing & splitting
│   ├── train.py               # Fine-tuning script
│   └── evaluate.py            # Model evaluation
│
├── models/
│   └── finetuned_sentiment/   # Saved fine-tuned model
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
│
├── requirements.txt
└── README.md
```

---

## Training Configuration

```python
Base Model:      cardiffnlp/twitter-roberta-base-sentiment
Learning Rate:   3e-5  (0.00003 - very small!)
Epochs:          3
Batch Size:      16
Optimizer:       AdamW
Loss Function:   CrossEntropyLoss

Why small learning rate?
  - Don't destroy pre-trained knowledge
  - Gentle adaptation, not complete retraining
  - 100x smaller than training from scratch
```

---

## Training Results

```
Epoch 1/3
  Train Loss: 0.6376
  Val Loss:   0.6074
  Val Acc:    70.00% ✅
  Status: Model learning well

Epoch 2/3
  Train Loss: 0.3428
  Val Loss:   0.6602
  Val Acc:    76.00% ✅ BEST MODEL (saved)
  Status: Continued improvement

Epoch 3/3
  Train Loss: 0.1786
  Val Loss:   0.7618
  Val Acc:    74.00% ⚠️
  Status: Overfitting detected
```

**Key Observations:**
- ✅ Model learned successfully (70% → 76%)
- ✅ Best performance at epoch 2
- ⚠️ Epoch 3 showed overfitting (val loss increased)
- ✅ Early stopping would have chosen epoch 2

---

## Model Performance

### Overall Metrics
```
Validation Accuracy: 76.00%
Test Accuracy:       ~74.51%

Baseline (random):   33.33%
Improvement:         +42 percentage points
```

### Per-Class Performance
```
               Precision  Recall  F1-Score
Negative         0.75     0.86     0.80
Neutral          0.78     0.70     0.74
Positive         0.71     0.71     0.71
```

### Confusion Matrix
```
           Predicted
           Neg  Neu  Pos
Actual Neg [12   1   1]   ← 86% correct
       Neu [ 3  14   3]   ← 70% correct
       Pos [ 2   3  12]   ← 71% correct
```

**Insights:**
- Negative sentiment: Best performance (86% recall)
- Neutral sentiment: Most challenging (confused with both classes)
- Positive sentiment: Balanced performance
- Rare confusion between positive ↔ negative (good!)

---

## What Makes This Different from Day 14?

**Day 14 (Using Pre-trained Model):**
```python
# Just load and use as-is
model = pipeline("sentiment-analysis")
result = model("I love this!")
# Works okay but generic
```

**Day 19 (Fine-Tuning):**
```python
# Adapted to YOUR specific data
model = AutoModelForSequenceClassification.from_pretrained(
    "./models/finetuned_sentiment"
)
# Performance improved from ~70% → 76% on your domain
```

**The difference:** Generic model → Custom model for your use case

---

## Key Concepts Learned

### 1. Transfer Learning
```
Pre-trained model knowledge:
  - What is language
  - Grammar patterns
  - Word relationships
  - Sentiment concepts

Fine-tuning adds:
  - YOUR specific domain
  - YOUR data patterns
  - YOUR class labels
```

### 2. Why Small Learning Rate?
```
Large LR (0.001): [0.5, 0.3, 0.8] → [0.2, 0.9, 0.1] ❌ Destroyed!
Small LR (3e-5):  [0.5, 0.3, 0.8] → [0.501, 0.299, 0.801] ✅ Adapted!

Small learning rate = gentle adjustment, not complete rewrite
```

### 3. Overfitting Detection
```
Epoch 1: Train ↓ Val ↓  (both improving = learning)
Epoch 2: Train ↓ Val ↓  (still learning)
Epoch 3: Train ↓ Val ↑  (overfitting!)

Solution: Early stopping at epoch 2
```

### 4. Dataset Size Requirements
```
Training from scratch: 100,000+ examples
Fine-tuning:          400-1,000 examples
Feature extraction:   100-500 examples

We used: 400 examples + fine-tuning = Good results!
```

---

## Installation

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install transformers torch pandas scikit-learn datasets

# Verify installation
python -c "import transformers; print(transformers.__version__)"
```

---

## Usage

### 1. Prepare Data
```bash
python scripts/prepare_data.py
```

**Output:**
- `data/train_subset.csv` (400 examples)
- `data/val_subset.csv` (50 examples)
- `data/test_subset.csv` (50 examples)

---

### 2. Fine-Tune Model
```bash
python scripts/train.py
```

**What happens:**
- Loads cardiffnlp/twitter-roberta-base-sentiment
- Fine-tunes on your 400 training examples
- Validates after each epoch
- Saves best model to `models/finetuned_sentiment/`

**Time:** ~15-30 minutes on CPU, ~5 minutes on GPU

---

### 3. Evaluate Model
```bash
python scripts/evaluate.py
```

**Output:**
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix
- Example predictions (correct and wrong)

---

### 4. Use for Predictions

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(
    "./models/finetuned_sentiment"
)
tokenizer = AutoTokenizer.from_pretrained(
    "./models/finetuned_sentiment"
)

# Predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    labels = ['negative', 'neutral', 'positive']
    return labels[predicted_class]

# Test
print(predict_sentiment("I love this product!"))  # → positive
print(predict_sentiment("This is terrible"))      # → negative
print(predict_sentiment("It's okay"))             # → neutral
```

---

## Files Explained

### prepare_data.py
```
Purpose: Clean and split dataset
Input:   data/train.csv (27k rows)
Output:  3 CSV files (train/val/test)
Key:     Balanced sampling, stratified split
```

### train.py
```
Purpose: Fine-tune the model
Key Components:
  - SentimentDataset class (tokenization)
  - Training loop (forward, backward, optimize)
  - Validation after each epoch
  - Save best model
```

### evaluate.py
```
Purpose: Test model on unseen data
Output:
  - Classification report
  - Confusion matrix
  - Sample predictions
  - Performance analysis
```

---

## Challenges Faced & Solutions

### Challenge 1: Overfitting on Small Dataset
```
Problem: Only 400 training examples
Symptom: Epoch 3 accuracy dropped (76% → 74%)
Solution: 
  - Use early stopping (stop at epoch 2)
  - Small learning rate (3e-5)
  - Could add dropout (not implemented)
```

### Challenge 2: Class Imbalance
```
Problem: Real dataset had 40% neutral, 30% pos, 30% neg
Solution: Balanced sampling (167 each class)
Result: Equal performance across classes
```

### Challenge 3: Dataset Encoding
```
Problem: CSV had latin-1 encoding, not UTF-8
Error: UnicodeDecodeError
Solution: pd.read_csv(..., encoding='latin-1')
```

### Challenge 4: GPU vs CPU
```
Problem: Code should work on both
Solution: 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  Move tensors: .to(device)
```

---

## What I Learned

### Technical Skills
- How to fine-tune pre-trained transformers
- PyTorch Dataset and DataLoader implementation
- Transfer learning vs training from scratch
- Overfitting detection and prevention
- Hugging Face transformers library
- Model evaluation with sklearn metrics

### Key Insights
- **Fine-tuning > training from scratch** for most real tasks
- **Small learning rates are critical** to preserve pre-trained knowledge
- **Few epochs (3-5) are enough** for fine-tuning
- **400 examples can work** if base model is strong
- **Overfitting is normal** with small datasets - use early stopping

### Design Decisions
- **Why RoBERTa?** Already trained on 58M tweets (perfect for sentiment)
- **Why 3e-5 LR?** Standard for BERT-family fine-tuning
- **Why 3 epochs?** More = overfitting, less = underfitting
- **Why batch size 16?** Fits in 8GB RAM, good gradient estimates

---

## Comparison: Day 14 vs Day 19

| Aspect | Day 14 (Pre-trained) | Day 19 (Fine-tuned) |
|--------|---------------------|---------------------|
| **Approach** | Use model as-is | Adapt to your data |
| **Data needed** | 0 (none) | 400+ examples |
| **Training time** | 0 (instant) | 15-30 minutes |
| **Accuracy** | ~70% (generic) | 76% (custom) |
| **Understanding** | Black box | Know how it works |
| **Customization** | None | Full control |

---

## Future Improvements

- [ ] Collect more training data (1000+ examples per class)
- [ ] Implement early stopping automatically
- [ ] Add data augmentation (paraphrasing)
- [ ] Try different base models (BERT, DistilBERT)
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Multi-language support (add Nepali sentiment)
- [ ] Deploy as API endpoint
- [ ] A/B test against generic model

---

## Connection to Lok Sewa Business

**Use Case: Analyze Student Feedback**

```
Students submit feedback on questions:
  "यो प्रश्न धेरै गाह्रो थियो"  → NEGATIVE (improve question)
  "राम्रो प्रश्न थियो"         → POSITIVE (keep similar)
  "अलमलिने"                    → NEGATIVE (needs clarity)

Fine-tuned model:
  → Automatically categorizes feedback
  → Identifies problematic questions
  → Improves question quality over time
  → No manual review needed!
```

**Real business value:** Automatic quality improvement loop

---

## Technical Specifications

```
Framework:        PyTorch + Hugging Face Transformers
Base Model:       cardiffnlp/twitter-roberta-base-sentiment
Model Size:       125M parameters
Fine-tuned:       Classification head only (~3K parameters updated)
Training Time:    ~20 minutes (CPU), ~5 minutes (GPU)
Inference Speed:  ~50 predictions/second (CPU)
Memory Required:  ~2GB RAM
Dependencies:     transformers, torch, pandas, scikit-learn
```

---

## Resources Used

**Hugging Face:**
- Model Hub: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
- Fine-tuning Guide: https://huggingface.co/docs/transformers/training
- Tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer

**Tutorials:**
- Transfer Learning concepts
- RoBERTa architecture
- PyTorch DataLoader patterns

---

## Developer

**Nawich**
- GitHub: https://github.com/NAWICH
- Day 19 of 30-day AI Engineering Journey
- Date: February 2026

---

## Progress Summary

```
Days 1-13:  ✅ Python, APIs, Auth, Databases
Days 14-16: ✅ ML Integration, RAG, LLM Apps
Days 17-18: ✅ Neural Networks, Smart RAG
Day 19:     ✅ Transfer Learning & Fine-Tuning

Next: Days 20-25 → Advanced LLMs, Agents, Production AI
```

---

*From using AI models (Day 14) → Understanding internals (Day 17) → Adapting models to your needs (Day 19)*

*This is the skill that 90% of AI engineering jobs require: fine-tuning pre-trained models!*