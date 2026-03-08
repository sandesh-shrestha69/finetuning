"""
Fine-tune sentiment model
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
#step 1: Configuration

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LEARNING_RATE = 3e-5  # ✅ Small for fine-tuning
EPOCHS = 3
BATCH_SIZE = 16

#step 2: Load Data
train_df = pd.read_csv("./data/train_subset.csv")
val_df = pd.read_csv("./data/val_subset.csv")

print(f"Train: {len(train_df)}, Val: {len(val_df)}")

#step 3: Create PyTorch Dataset
class SentimentDataset(Dataset):
    """
    __init__: Store texts, labels, tokenizer
    __len__: Return number of examples
    __getitem__: Return tokenized text + label for one example
    """
    
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
            padding = 'max_length',
            truncation = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label,dtype=torch.long)
        }


#step 4: Prepare Data

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Convert sentiment strings to numbers
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_labels = train_df['sentiment'].map(label_map).tolist()
val_labels = val_df['sentiment'].map(label_map).tolist()

# Create datasets
train_dataset = SentimentDataset(
    texts=train_df['text'].tolist(),
    labels=train_labels,
    tokenizer=tokenizer
)

val_dataset = SentimentDataset(
    texts=val_df['text'].tolist(),
    labels=val_labels,
    tokenizer=tokenizer
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# STEP 5: Load Model and Optimizer
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3  # negative, neutral, positive
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# STEP 6: Training Loop
best_val_acc = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(EPOCHS):
    # ─── Training Phase ───
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        # YOUR CODE:
        # 1. Get input_ids, attention_mask, labels from batch
        # 2. Forward pass: outputs = model(...)
        # 3. Get loss: loss = outputs.loss
        # 4. Backward: optimizer.zero_grad(), loss.backward(), optimizer.step()
        # 5. Track loss
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        output = model(input_ids = input_ids, attention_mask= attention_mask, labels = labels)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    
    # ─── Validation Phase ───
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # YOUR CODE:
            # 1. Get inputs and labels
            # 2. Forward pass
            # 3. Get predictions (argmax of logits)
            # 4. Count correct predictions
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim = 1)

            correct +=(preds == labels).sum().item()
            total += labels.size(0)
            pass
    
    # Calculate metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total * 100
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("./models/finetuned_sentiment")
        tokenizer.save_pretrained("./models/finetuned_sentiment")
        print(f" Saved (best: {best_val_acc:.2f}%)")

print(f"\nraining complete! Best accuracy: {best_val_acc:.2f}%")