import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertModel,
    DataCollatorWithPadding,
    get_scheduler,
)
from datasets import load_dataset
from tqdm import tqdm

# === Model Definition ===
class BertRegressionModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim=128, num_models=1):
        super().__init__()
        self.bert   = BertModel.from_pretrained(model_name)
        self.cls    = nn.Linear(config.hidden_size, hidden_dim)
        self.relu   = nn.ReLU()
        self.fc1    = nn.Linear(hidden_dim + num_models, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, model_onehot):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls  = out.last_hidden_state[:, 0, :]
        h1   = self.relu(self.cls(cls))
        x    = torch.cat([h1, model_onehot], dim=1)
        h2   = self.relu(self.fc1(x))
        return self.fc2(h2).squeeze(-1)


def main():
    class Args:
        def __init__(self):
            self.model_name = "bert-base-uncased"
            self.batch_size = 16
            self.num_epochs = 8
            self.lr = 1e-5
            self.max_samples = 50000
            self.patience = 2

    args = Args()

    # global tokenizer for preprocess()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1) Load & sample dataset ──
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        trust_remote_code=True
    ).shuffle(seed=42).select(range(args.max_samples))

    # ── 2) Determine model_names & save order ──
    feat = ds.features["model"]
    if hasattr(feat, "names"):
        model_names = feat.names
    else:
        model_names = sorted(set(ds["model"]))
    num_models = len(model_names)
    print(f"Using {num_models} LLMs in order:\n{model_names}")
    with open("model_names.json","w") as fp:
        json.dump(model_names, fp, indent=2)

    # ── 3) Preprocess & filter ──
    def preprocess(example):
        conv = example["conversation"]
        if isinstance(conv, list) and len(conv) >= 2:
            prompt, response = conv[0]["content"], conv[1]["content"]
            tok = tokenizer(prompt, truncation=True,
                            padding="max_length", max_length=256)
            resp_ids = tokenizer(response, truncation=True,
                                 padding=False)["input_ids"]
            tok["labels"] = len(resp_ids)
            # one-hot for this example’s model
            oh = [1 if example["model"]==m else 0 for m in model_names]
            tok["model_onehot"] = oh
            return tok
        tok = tokenizer("", truncation=True,
                        padding="max_length", max_length=256)
        tok["labels"]       = -1
        tok["model_onehot"] = [0]*num_models
        return tok

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: ex["labels"] > 0)
    print("After filtering:", len(ds), "samples")

    # ── 4) Train/Val split ──
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── 5) Dataloaders ──
    base_collator = DataCollatorWithPadding(tokenizer)
    def collate_fn(exs):
        ohs  = [e.pop("model_onehot") for e in exs]
        labs = [e.pop("labels")        for e in exs]
        batch = base_collator(exs)
        batch["model_onehot"] = torch.tensor(ohs,  dtype=torch.float32)
        batch["labels"]       = torch.tensor(labs, dtype=torch.float32)
        return batch

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # ── 6) Build model, optimizer, scheduler, loss ──
    config    = AutoConfig.from_pretrained(args.model_name)
    model     = BertRegressionModel(config, args.model_name,
                                    hidden_dim=128,
                                    num_models=num_models
                                   ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.num_epochs
    scheduler   = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=0,
                                num_training_steps=total_steps)
    criterion = nn.L1Loss()

    # ── 7) Training with early stopping ──
    best_val_loss = float("inf")
    patience_cnt  = 0
    best_path     = "best_length_model.pth"

    for epoch in range(1, args.num_epochs+1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            onehot         = batch["model_onehot"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, onehot)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # — evaluate on validation set —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                iids = batch["input_ids"].to(device)
                am   = batch["attention_mask"].to(device)
                oh   = batch["model_onehot"].to(device)
                labs = batch["labels"].to(device)
                preds = model(iids, am, oh)
                val_loss += criterion(preds, labs).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} — Train L1: {train_loss:.4f}  Val L1: {val_loss:.4f}")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.to("cpu").state_dict(), best_path,
                       _use_new_zipfile_serialization=False)
            model.to(device)
            print(f"  ↳ New best, saved to {best_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Stopping early (no val improvement for {args.patience} epochs).")
                break

    print("Done training. Best Val L1:", best_val_loss)


if __name__ == "__main__":
    main()