import os
os.environ["GRPC_VERBOSITY"] = "NONE"

import gc
import math
import json
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna

seed_value = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

def fix_seeds(seed_value: int = seed_value, device: str = "cpu") -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_pe(max_len), persistent=False)

    def _build_pe(self, length):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, length, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        if seq_len > self.pe.size(1):
            pe_big = self._build_pe(seq_len).to(x.device)
            return x + pe_big
        else:
            return x + self.pe[:, :seq_len, :]

class ICDMLMModel(nn.Module):
    def __init__(self, num_codes, embed_dim, num_heads, hidden_dim, num_layers, padding_idx, dropout, output_num_layers, max_len):
        super(ICDMLMModel, self).__init__()
        self.embedding = nn.Embedding(num_codes, embed_dim, padding_idx)
        
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.embedding_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        if output_num_layers == 1:
            self.output_layer = nn.Linear(embed_dim, num_codes)
        else:
            if output_num_layers == 2:
                self.output_layer = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_codes)
                )
            else:
                self.output_layer = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_codes)
                )

        self.apply(self._init_weights)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.embedding_dropout(x)
        x = self.transformer(x)
        logits = self.output_layer(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

class MaskedICDDataset(Dataset):
    def __init__(self, sequences, vocab, max_seq_len=100, mask_prob=0.15):
        self.sequences = sequences
        self.vocab     = vocab
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_id   = vocab['[MASK]']
        self.pad_id    = vocab['[PAD]']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_seq_len]
        ids = [self.vocab[c] for c in seq]
        pad = [self.pad_id] * (self.max_seq_len - len(ids))
        input_ids = ids + pad

        labels = [-100] * self.max_seq_len

        rng = random.Random(seed_value + idx)

        for i in range(len(ids)):
            if rng.random() < self.mask_prob:
                labels[i] = input_ids[i]
                r = rng.random()      
                if r < 0.8:
                    input_ids[i] = self.mask_id
                elif r < 0.9:
                    input_ids[i] = random.choice(list(self.vocab.values()))
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels':    torch.tensor(labels,    dtype=torch.long)
        }
    
def calculate_accuracy(logits, labels, ignore_index=-100):
    preds = logits.argmax(dim=-1)
    mask  = labels != ignore_index
    correct = (preds[mask] == labels[mask]).sum().float()
    total   = mask.sum().float()
    return (correct / total).item() if total > 0 else 0.0

def train_model(model, train_dataloader, val_dataloader, vocab_size, epochs, 
                lr, weight_decay, scheduler=None, device="cuda", patience=3, epsilon=1e-3, trial=None):
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_loss = float('inf')
    no_improve_count = 0
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_accuracy = 0
        
        for batch in train_dataloader:
            masked_inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            
            logits = model(masked_inputs)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            accuracy = calculate_accuracy(logits, labels)
            train_accuracy += accuracy
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = train_accuracy / len(train_dataloader)
        
        model.eval()
        total_val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                masked_inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)

                logits = model(masked_inputs)
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
                total_val_loss += loss.item()
                
                accuracy = calculate_accuracy(logits, labels)
                val_accuracy += accuracy
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = val_accuracy / len(val_dataloader)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_accuracy:.4f}| "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_accuracy:.4f}")

        val_losses.append(avg_val_loss)
        if scheduler:
            scheduler.step(avg_val_loss)
    
        if trial is not None:
            trial.report(avg_val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if best_val_loss - avg_val_loss > epsilon:
            best_val_loss = avg_val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break
    
    return best_val_loss

def objective(trial, train_dataloader, val_dataloader, max_len, vocab_size):
    if torch.cuda.is_available():
        gpu_id = trial.number % torch.cuda.device_count()
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    fix_seeds(seed_value, device)
    print(f"Trial {trial.number} using {device}")
        
    lr = trial.suggest_categorical("lr", [5e-5, 1e-4, 3e-4, 5e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-3, 1e-2])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.2])
    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 7)
    num_heads_candidates = [h for h in range(1, 9) if embed_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", num_heads_candidates)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    output_num_layers = trial.suggest_int("output_num_layers", 1, 3)

    scheduler_name = "ReduceLROnPlateau"
    padding_idx = 0
    num_epochs = 100
    patience = 3
            
    print("loaded data")

    model = ICDMLMModel(
        num_codes=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        padding_idx=padding_idx,
        dropout=dropout,
        output_num_layers=output_num_layers,
        max_len=max_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"param number: {total_params:,}")     

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), 
            epochs=num_epochs, anneal_strategy='linear'
        )
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=2, factor=0.5
        )
    else:
        scheduler = None

    try:
        val_loss = train_model(
            model,
            train_dataloader,
            val_dataloader,
            vocab_size=vocab_size,
            epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            scheduler=scheduler,
            patience=patience,
            device=device,
            trial=trial,
        )
    except optuna.exceptions.TrialPruned:
        del model, optimizer, scheduler
        del train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()
        raise
    
    
    del model, optimizer, scheduler
    del  train_dataloader, val_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    
    print(f"validation loss: {val_loss:.4f}")
    
    return val_loss


def main():
    fix_seeds(seed_value, device)
    parser = argparse.ArgumentParser(description="MLM HPO")
    parser.add_argument("--input", required=True, help='Input file')
    parser.add_argument("--output", required=True, help='Output file')
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    df = df.sort_values(["subject_id", "admittime", "seq_num"])
    df["seq_num"] = df.groupby("subject_id").cumcount() + 1

    unique_icds = df["category"].unique()
    PAD_TOKEN, PAD_TOKEN_ID = "[PAD]", 0
    icd2id = {code: idx for idx, code in enumerate(unique_icds, start=1)}
    icd2id[PAD_TOKEN] = PAD_TOKEN_ID

    MASK_TOKEN, MASK_TOKEN_ID = "[MASK]", len(icd2id)
    icd2id[MASK_TOKEN] = MASK_TOKEN_ID

    vocab_size = len(icd2id)
    df["icd_id"] = df["category"].map(icd2id)
    df = df[["subject_id", "seq_num", "category", "icd_id"]]

    all_subjects = df['subject_id'].unique()
    np.random.shuffle(all_subjects)
    split_idx = int(0.8 * len(all_subjects))
    train_subjects, val_subjects = all_subjects[:split_idx], all_subjects[split_idx:]

    sequences_df = df.groupby('subject_id')['category'].apply(list)
    train_sequences = sequences_df.loc[train_subjects].tolist()
    val_sequences = sequences_df.loc[val_subjects].tolist()

    train_seq_idx = [[code for code in seq] for seq in train_sequences]
    val_seq_idx = [[code for code in seq] for seq in val_sequences]

    max_len = 100
    batch_size = 128

    train_dataset = MaskedICDDataset(train_seq_idx, icd2id, max_seq_len=max_len)
    val_dataset = MaskedICDDataset(val_seq_idx, icd2id, max_seq_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    study = optuna.create_study(direction="minimize", pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15))

    study.enqueue_trial({
        "lr": 0.001,
        "weight_decay": 0.001,
        "dropout": 0.1,
        "embed_dim": 256, # no default
        "num_layers": 4, # no default
        "num_heads": 4, # no default
        "hidden_dim": 1024, # 2048
        "output_num_layers": 1,
    })
    print("start trials")

    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader, max_len, vocab_size), n_trials=100, gc_after_trial=True)

    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda x: x.value
    )

    with open(args.output.replace(".json", "all_trials.json"), "w") as f:
        results = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params
            }
            for t in sorted_trials
        ]
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()