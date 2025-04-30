import os
import random
import torch
import numpy as np
import pandas as pd
import csv
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast, get_linear_schedule_with_warmup

csv.field_size_limit(sys.maxsize)

# Configuration
class Config:
    seed = 42
    model_name = 'bert-base-uncased'
    num_labels = 2
    dropout_prob = 0.3
    max_len = 128
    batch_size = 32
    lr = 2e-5
    epochs = 3
    # because i train on a mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_file = 'trainingData/train.csv'
    test_file = 'testingData/test.csv'
    ckpt_dir = Path('checkpoints')

cfg = Config()

def set_seed(seed=None):
    """Set random seed for reproducibility."""
    seed = seed or cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, tokenizer, epoch):
    """Save model and tokenizer to a checkpoint directory."""
    ckpt_dir = cfg.ckpt_dir / f"withContent_epoch_{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Saved checkpoint to {ckpt_dir}")

def load_checkpoint(model_dir):
    """Load model and tokenizer from a checkpoint directory."""
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model.to(cfg.device)
    model.eval()
    return model, tokenizer

def load_split(path):
    """Load data from CSV file."""
    records = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            title = row.get("title")
            content = row.get("content")
            cls = row.get("classification")
            label = 0 if cls == "fake" else 1
            records.append({"title": title, "content": content, "label": label})
    return pd.DataFrame(records)

def load_train():
    """Load training data."""
    return load_split(cfg.train_file)

def load_test():
    """Load test data."""
    return load_split(cfg.test_file)

# dataset Class  from this doc https://pytorch.org/tutorials/beginner/basics/data_tutorial.html cause it kill the performance otherwise
class NewsDataset(Dataset):
    """PyTorch Dataset for news classification using title and content."""
    
    def __init__(self, dataframe, tokenizer=None, max_len=None):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer or BertTokenizerFast.from_pretrained(cfg.model_name)
        self.max_len = max_len or cfg.max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Combine title and content with [SEP] token
        text = row['title'] + ' ' + self.tokenizer.sep_token + ' ' + row['content']
        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }

# Model Functions
def get_tokenizer():
    """Get a pre-trained BERT tokenizer."""
    return BertTokenizerFast.from_pretrained(cfg.model_name)

def get_model():
    """Get a pre-trained BERT model for sequence classification."""
    model = BertForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        hidden_dropout_prob=cfg.dropout_prob
    )
    return model.to(cfg.device)

def initialize_training():
    """Initialize everything needed for training."""
    set_seed()
    
    model = get_model()
    tokenizer = get_tokenizer()

    train_df = load_train()
    test_df = load_test()

    max_len = cfg.max_len * 2
    
    train_dataset = NewsDataset(train_df, tokenizer, max_len=max_len)
    val_dataset = NewsDataset(test_df, tokenizer, max_len=max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size, 
        shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    
    total_steps = len(train_loader) * cfg.epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    return model, tokenizer, train_loader, val_loader, optimizer, scheduler, cfg.device

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()
    
    losses = []
    
    for batch in tqdm(train_loader, desc="Train Epoch"):
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )
        
        loss = outputs.loss
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step()
        
        losses.append(loss.item())
    
    return sum(losses) / len(losses)

def validate(model, val_loader, device):
    """Evaluate model on validation data."""
    model.eval()
    
    preds, golds = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate"):
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            ).logits
            
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            
            preds.extend(batch_preds)
            golds.extend(batch['labels'].tolist())
    
    return accuracy_score(golds, preds), f1_score(golds, preds)

def run_training(epochs=None):
    """Run the full training process."""
    epochs = epochs or cfg.epochs

    model, tokenizer, train_loader, val_loader, optimizer, scheduler, device = initialize_training()
    
    best_f1 = 0.0
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        acc, f1 = validate(model, val_loader, device)
        
        print(f"Epoch {epoch} | loss {train_loss:.4f} | acc {acc:.4f} | f1 {f1:.4f}")
        
        # Save only if better
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, tokenizer, epoch)

def evaluate_model(model_path, test_file=None):
    """Evaluate a trained model on test data."""
    model, tokenizer = load_checkpoint(model_path)
    
    if test_file:
        test_df = load_split(test_file)
    else:
        test_df = load_test()
    
    max_len = cfg.max_len * 2
    
    test_dataset = NewsDataset(test_df, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    device = cfg.device
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"Modele: {model_path}")
    print(f"Test samples: {len(test_df)}")
    print(f"Précision: {accuracy:.4f}")
    print(f"Score F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))
    print("\nMatrice de confusion:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

print("Training...")
run_training() 
print("Training finished")
