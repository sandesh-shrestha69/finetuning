"""
Evaluate fine-tuned sentiment model on test set
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

# SentimentDataset class (copied from train.py)
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load fine-tuned model
model_path = "./models/finetuned_sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"✅ Loaded model from {model_path}")

# 2. Load test data
test_df = pd.read_csv("./data/test_subset.csv")
print(f"✅ Loaded {len(test_df)} test examples")

# 3. Convert sentiment to numbers
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
test_labels = test_df['sentiment'].map(label_map).tolist()

# 4. Create test dataset
test_dataset = SentimentDataset(
    texts=test_df['text'].tolist(),
    labels=test_labels,
    tokenizer=tokenizer
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Evaluation
print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 6. Calculate metrics
print("\n" + "="*60)
print("RESULTS")
print("="*60)

accuracy = accuracy_score(all_labels, all_preds)
print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=['negative', 'neutral', 'positive'],
    digits=4
))

print("\n🔢 Confusion Matrix:")
print("           Predicted")
print("           Neg  Neu  Pos")
cm = confusion_matrix(all_labels, all_preds)
for i, row in enumerate(cm):
    label = ['Neg', 'Neu', 'Pos'][i]
    print(f"Actual {label} {row}")

# 7. Show some example predictions
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

reverse_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Show 3 correct and 3 wrong predictions
correct_examples = []
wrong_examples = []

for i in range(len(all_labels)):
    if all_labels[i] == all_preds[i]:
        correct_examples.append(i)
    else:
        wrong_examples.append(i)

print("\n✅ CORRECT Predictions (first 3):")
for idx in correct_examples[:3]:
    text = test_df.iloc[idx]['text'][:80]
    true_label = reverse_map[all_labels[idx]]
    pred_label = reverse_map[all_preds[idx]]
    print(f"  Text: {text}...")
    print(f"  True: {true_label} | Predicted: {pred_label}\n")

print("❌ WRONG Predictions (first 3):")
for idx in wrong_examples[:3]:
    text = test_df.iloc[idx]['text'][:80]
    true_label = reverse_map[all_labels[idx]]
    pred_label = reverse_map[all_preds[idx]]
    print(f"  Text: {text}...")
    print(f"  True: {true_label} | Predicted: {pred_label}\n")

print("="*60)
print(f"🎉 Evaluation complete!")
print(f"Best validation accuracy during training: 76.00%")
print(f"Test set accuracy: {accuracy*100:.2f}%")
print("="*60)