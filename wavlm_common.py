import argparse
import csv
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
CROP_SECONDS = 3.0
DATASET_NAME = "saeedzou/common-voice-17-en-age-gender-accent-sampled"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--dataset_repo", default=DATASET_NAME, help="Hugging Face dataset repository to load")
    parser.add_argument("--layer", type=int, default=-1, help="hidden_states index, -1 for last")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--clf_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clf_batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--metrics_csv", default=None, help="path to save per-epoch and final metrics as CSV")
    return parser


def add_classification_loss_args(parser):
    parser.add_argument("--loss", choices=("ce", "focal"), default="ce", help="classification loss to use")
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="focusing parameter used when --loss focal is selected",
    )
    return parser


def load_wavlm(model_name, device):
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device).eval()
    return extractor, model


def load_split(split, max_samples=None, dataset_repo=DATASET_NAME):
    ds = load_dataset(dataset_repo, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def get_label_list(dataset, column):
    labels = {x for x in dataset[column] if x not in (None, "")}
    return sorted(labels)


def filter_labeled(dataset, column):
    return dataset.filter(lambda x: x[column] not in (None, ""))


def build_label_mapping(labels):
    label_list = sorted(set(labels))
    label2id = {label: index for index, label in enumerate(label_list)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def filter_to_labels(dataset, column, allowed_labels):
    allowed = set(allowed_labels)
    return dataset.filter(lambda x: x[column] in allowed)


def crop_or_pad(waveform, sample_idx=0, epoch_idx=0, seed=42, seconds=CROP_SECONDS, sr=SAMPLE_RATE):
    length = int(sr * seconds)
    n = waveform.shape[-1]
    if n >= length:
        # Generate a deterministic chunk seed for epoch n and utterance m
        chunk_seed = hash((seed, epoch_idx, sample_idx)) % (2**31 - 1)
        rng = random.Random(chunk_seed)
        start = rng.randrange(0, n - length + 1)
        return waveform[start:start + length]
    return F.pad(waveform, (0, length - n))


def load_waveform(audio, sample_idx=0, epoch_idx=0, seed=42, sr=SAMPLE_RATE):
    waveform = torch.tensor(audio["array"], dtype=torch.float32)
    audio_sampling_rate = audio["sampling_rate"]
    if audio_sampling_rate != sr:
        waveform = torchaudio.functional.resample(waveform, audio_sampling_rate, sr)
    return crop_or_pad(waveform, sample_idx=sample_idx, epoch_idx=epoch_idx, seed=seed, sr=sr)


@torch.no_grad()
def embed_batch_all_layers(waveforms, extractor, model, device):
    inputs = extractor(
        [w.numpy() for w in waveforms],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = model(input_values)

    # outputs.hidden_states contains (num_layers + 1) tensors
    pooled_layers = []
    for hidden in outputs.hidden_states:
        feat_mask = model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        feat_mask = feat_mask.unsqueeze(-1).float()
        pooled = (hidden * feat_mask).sum(1) / feat_mask.sum(1).clamp(min=1)
        pooled_layers.append(pooled.cpu().numpy())  # Shape: [batch_size, hidden_dim]

    # Stack along layer dimension -> Shape: [num_layers, batch_size, hidden_dim]
    return np.stack(pooled_layers, axis=0)


# --- Embedding Precomputation & Disk Caching ---

def compute_chunk_variations(
    dataset, label_column, label2id, extractor, model, device, batch_size, seed, num_chunks=3
):
    num_samples = len(dataset)
    labels = np.array([label2id[label] for label in dataset[label_column]])

    X_chunks = []
    for chunk_idx in range(num_chunks):
        X_batches = []
        for i in tqdm(
            range(0, num_samples, batch_size),
            desc=f"Extracting all layers | chunk {chunk_idx + 1}/{num_chunks}"
        ):
            batch = dataset[i:i + batch_size]
            waveforms = [
                load_waveform(audio, sample_idx=i + idx, epoch_idx=chunk_idx, seed=seed)
                for idx, audio in enumerate(batch["audio"])
            ]
            # Shape: [num_layers, batch_size, hidden_dim]
            emb = embed_batch_all_layers(waveforms, extractor, model, device)
            X_batches.append(emb)

        ep_embs = np.concatenate(X_batches, axis=1)  # [num_layers, num_samples, hidden_dim]
        X_chunks.append(ep_embs)

    X_all = np.stack(X_chunks, axis=0)  # [num_chunks, num_layers, num_samples, hidden_dim]
    return X_all, labels


def load_embeddings(
    dataset,
    split_name,
    label_column,
    label2id,
    model_name,
    layer,
    device,
    batch_size,
    seed=42,
    num_chunks=3,
    cache_dir="./embeddings_cache",
):
    cache_path = Path(cache_dir)
    clean_model = model_name.replace("/", "_")

    # Folder for this split: embeddings_cache/emb_microsoft_wavlm-base-plus_train_seed42_chunks3_N1256/
    split_dir = cache_path / f"emb_{clean_model}_{split_name}_seed{seed}_chunks{num_chunks}_N{len(dataset)}"
    split_dir.mkdir(parents=True, exist_ok=True)

    layer_file = split_dir / f"layer_{layer}.pt"

    # 1. Fast path: load ONLY the requested layer file from disk
    if layer_file.exists():
        print(f"Loading precomputed embeddings for layer {layer} from: {layer_file}")
        cached = torch.load(layer_file, weights_only=False)
        return cached["X"], cached["y"]

    # 2. Compute path: extract all layers once and save each layer into its own .pt file
    print(f"Precomputing embeddings for ALL layers of '{split_name}' ({num_chunks} chunk variations)...")
    extractor, model = load_wavlm(model_name, device)
    X_all, y = compute_chunk_variations(
        dataset, label_column, label2id, extractor, model, device, batch_size, seed, num_chunks
    )
    # X_all shape: [num_chunks, num_layers, num_samples, hidden_dim]

    num_layers = X_all.shape[1]
    print(f"Saving per-layer embeddings into folder: {split_dir}")

    # Save layer_0.pt, layer_1.pt, ..., layer_12.pt
    for l_idx in range(num_layers):
        l_file = split_dir / f"layer_{l_idx}.pt"
        torch.save({"X": X_all[:, l_idx], "y": y}, l_file)

    # Save layer_-1.pt as an alias for the final layer
    torch.save({"X": X_all[:, -1], "y": y}, split_dir / "layer_-1.pt")

    X_layer = X_all[:, layer]
    return X_layer, y

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class ResidualMLP(torch.nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512, num_blocks=3, dropout=0.3):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, hidden_dim)
        self.blocks = torch.nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(num_blocks)]
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(self.norm(x))
        return self.out(x)


def safe_stratified_split(X, y, test_size, random_state):
    class_counts = np.bincount(y)
    if class_counts.size == 0:
        raise ValueError("Cannot split empty training data.")
    if class_counts.min() < 2:
        raise ValueError(
            "Stratified validation split requires at least 2 training examples per class. "
            "Increase the sample size or lower the number of classes in the training subset."
        )
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as exc:
        raise ValueError(
            f"Unable to create a stratified validation split with val_ratio={test_size}. "
            "Try a larger training set or a smaller validation ratio."
        ) from exc


TRAINING_METRICS_FIELDNAMES = [
    "run_id",
    "task",
    "stage",
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
    "epoch_time_s",
    "eta_s",
    "best_epoch",
    "best_val_acc",
    "test_acc",
    "classification_report",
    "model",
    "layer",
    "seed",
    "loss",
    "focal_gamma",
    "device",
]


def get_metrics_csv_path(task_name, args):
    custom_path = getattr(args, "metrics_csv", None)
    if custom_path:
        return Path(custom_path)
    metrics_dir = Path(__file__).resolve().parent / "metrics"
    return metrics_dir / f"{task_name}_metrics.csv"


def save_training_metrics_csv(metrics_path, rows):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# --- Deterministic Training & Batching ---

def train_eval(X_train, y_train, X_val, y_val, X_test, y_test, id2label, args, task_name):
    device = args.device
    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Ensure inputs are 3D tensors: [num_epochs, num_samples, hidden_dim]
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=0)
    if X_val.ndim == 2:
        X_val = np.expand_dims(X_val, axis=0)
    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, axis=0)

    # Fit scaler on epoch 0 chunk embeddings
    scaler = StandardScaler().fit(X_train[0])
    X_val_scaled = scaler.transform(X_val[0])
    X_test_scaled = scaler.transform(X_test[0])

    val_ds = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    val_loader = DataLoader(val_ds, batch_size=args.clf_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.clf_batch_size, shuffle=False)

    num_classes = len(id2label)
    model = ResidualMLP(X_train.shape[-1], num_classes, args.hidden_dim, args.num_blocks, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.clf_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def compute_loss(logits, targets):
        if args.loss == "focal":
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            return ((1.0 - pt) ** args.focal_gamma * ce_loss).mean()
        return F.cross_entropy(logits, targets)

    best_val_acc, best_state, patience_left = 0.0, None, args.patience
    best_epoch = None
    epoch_rows = []
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()

        # Get precomputed embeddings for current chunk index: chunk[epoch][m]
        ep_idx = epoch % X_train.shape[0]
        X_train_ep = scaler.transform(X_train[ep_idx])

        # Seeding PyTorch DataLoader shuffling ensures identical batch compositions across runs
        g = torch.Generator()
        g.manual_seed(args.seed + epoch)

        train_ds = TensorDataset(torch.tensor(X_train_ep, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=args.clf_batch_size, shuffle=True, generator=g)

        train_loss_sum = 0.0
        train_examples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = compute_loss(logits, yb)
            loss.backward()
            optimizer.step()
            batch_size = yb.size(0)
            train_loss_sum += loss.item() * batch_size
            train_examples += batch_size
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss = compute_loss(logits, yb)
                preds = logits.argmax(dim=1)
                batch_size = yb.size(0)
                val_loss_sum += val_loss.item() * batch_size
                val_correct += (preds == yb).sum().item()
                val_total += batch_size
        val_acc = val_correct / val_total
        train_loss = train_loss_sum / max(train_examples, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_elapsed = (time.perf_counter() - start_time) / (epoch + 1)
        remaining_seconds = avg_elapsed * (args.epochs - epoch - 1)
        eta_minutes, eta_seconds = divmod(int(round(remaining_seconds)), 60)
        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f} | "
            f"epoch time {epoch_elapsed:.1f}s | eta {eta_minutes:02d}:{eta_seconds:02d}"
        )

        if val_acc > best_val_acc:
            best_val_acc, best_state, patience_left = val_acc, {k: v.clone() for k, v in model.state_dict().items()}, args.patience
            best_epoch = epoch + 1
        else:
            patience_left -= 1

        epoch_rows.append(
            {
                "run_id": run_id,
                "task": task_name,
                "stage": "epoch",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_s": epoch_elapsed,
                "eta_s": remaining_seconds,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_acc": None,
                "classification_report": None,
                "model": args.model,
                "layer": args.layer,
                "seed": args.seed,
                "loss": args.loss,
                "focal_gamma": args.focal_gamma,
                "device": device,
            }
        )

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    acc = accuracy_score(all_labels, all_preds)
    labels = sorted(id2label)
    target_names = [id2label[i] for i in labels]
    report = classification_report(all_labels, all_preds, labels=labels, target_names=target_names, zero_division=0)

    metrics_rows = [
        *epoch_rows,
        {
            "run_id": run_id,
            "task": task_name,
            "stage": "final",
            "epoch": None,
            "train_loss": None,
            "val_loss": None,
            "val_acc": None,
            "epoch_time_s": None,
            "eta_s": None,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc": acc,
            "classification_report": report,
            "model": args.model,
            "layer": args.layer,
            "seed": args.seed,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "device": device,
        },
    ]
    metrics_path = get_metrics_csv_path(task_name, args)
    save_training_metrics_csv(metrics_path, metrics_rows)
    print(f"saved metrics csv to {metrics_path}")
    return acc, report, metrics_path


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denominator = np.clip(denominator, a_min=1e-12, a_max=None)
    return numerator / denominator


def sample_verification_trials(embeddings_by_speaker, pos_pairs_per_speaker=5, neg_pairs_per_speaker=5, seed=42):
    rng = random.Random(seed)
    speakers = [speaker for speaker, feats in embeddings_by_speaker.items() if len(feats) >= 2]
    if len(speakers) < 2:
        raise ValueError("Need at least two speakers with two or more utterances each for verification trials.")

    pairs = []
    labels = []

    for speaker in speakers:
        feats = embeddings_by_speaker[speaker]
        positive_pairs = list(combinations(range(len(feats)), 2))
        if len(positive_pairs) > pos_pairs_per_speaker:
            positive_pairs = rng.sample(positive_pairs, pos_pairs_per_speaker)
        for left_index, right_index in positive_pairs:
            pairs.append((feats[left_index], feats[right_index]))
            labels.append(1)

    for speaker in speakers:
        left_feats = embeddings_by_speaker[speaker]
        other_speakers = [candidate for candidate in speakers if candidate != speaker]
        if not other_speakers:
            continue
        available = min(neg_pairs_per_speaker, len(left_feats))
        left_indices = rng.sample(range(len(left_feats)), available)
        for left_index in left_indices:
            other_speaker = rng.choice(other_speakers)
            right_feats = embeddings_by_speaker[other_speaker]
            right_index = rng.randrange(len(right_feats))
            pairs.append((left_feats[left_index], right_feats[right_index]))
            labels.append(0)

    if not pairs:
        raise ValueError("No verification trials could be formed from the provided speaker embeddings.")

    left = np.stack([pair[0] for pair in pairs], axis=0)
    right = np.stack([pair[1] for pair in pairs], axis=0)
    return left, right, np.asarray(labels, dtype=np.int64)


def verification_metrics(left_embeddings, right_embeddings, labels):
    scores = cosine_similarity(left_embeddings, right_embeddings)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    threshold = float(thresholds[eer_index])
    predictions = (scores >= threshold).astype(np.int64)
    accept_accuracy = float((predictions == labels).mean())
    reject_rate = float(((predictions == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1))
    return {
        "roc_auc": float(auc),
        "eer": eer,
        "threshold": threshold,
        "accept_accuracy": accept_accuracy,
        "reject_rate": reject_rate,
    }
