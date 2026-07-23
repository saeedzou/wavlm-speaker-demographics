import argparse
import hashlib
import json
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
DATASET_NAME = "saeedzou/common-voice-17-en-age-gender-sampled"
EMBEDDING_CACHE_VERSION = 1
DEFAULT_EMBEDDING_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "wavlm_embeddings"


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
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=None)
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
    parser.add_argument("--cache_dir", default=str(DEFAULT_EMBEDDING_CACHE_DIR), help="directory used to cache embeddings")
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


def _dataset_fingerprint(dataset):
    return getattr(dataset, "_fingerprint", None) or f"len-{len(dataset)}"


def _embedding_cache_payload(dataset, label_column, model_name, layer, device, seed):
    return {
        "version": EMBEDDING_CACHE_VERSION,
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "dataset_size": len(dataset),
        "label_column": label_column,
        "model_name": model_name,
        "layer": layer,
        "device": device,
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "crop_seconds": CROP_SECONDS,
    }


def get_embeddings_cache_path(dataset, label_column, model_name, layer, device, seed, cache_dir=None):
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_EMBEDDING_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = _embedding_cache_payload(dataset, label_column, model_name, layer, device, seed)
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_root / f"{digest}.npz"


def load_cached_embeddings(cache_path):
    if cache_path is None or not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cached:
        return cached["X"], cached["y"]


def save_cached_embeddings(cache_path, X, y):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)


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


def crop_or_pad(waveform, seconds=CROP_SECONDS, sr=SAMPLE_RATE, seed=None, key=None):
    length = int(sr * seconds)
    n = waveform.shape[-1]
    if n >= length:
        if seed is None or key is None:
            start = random.randint(0, n - length)
        else:
            digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
            start = int.from_bytes(digest, byteorder="big") % (n - length + 1)
        return waveform[start:start + length]
    return F.pad(waveform, (0, length - n))


def load_waveform(example, sr=SAMPLE_RATE, seed=None):
    audio = example["audio"]
    waveform = torch.tensor(audio["array"], dtype=torch.float32)
    if audio["sampling_rate"] != sr:
        waveform = torchaudio.functional.resample(waveform, audio["sampling_rate"], sr)
    crop_key = audio.get("path")
    if crop_key is None:
        crop_key = hashlib.sha1(waveform.numpy().tobytes()).hexdigest()
    return crop_or_pad(waveform, sr=sr, seed=seed, key=crop_key)


@torch.no_grad()
def embed_batch(waveforms, extractor, model, layer, device):
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
    hidden = outputs.hidden_states[layer]
    feat_mask = model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
    feat_mask = feat_mask.unsqueeze(-1).float()
    pooled = (hidden * feat_mask).sum(1) / feat_mask.sum(1).clamp(min=1)
    return pooled.cpu().numpy()


def compute_embeddings(dataset, label_column, label2id, extractor, model, layer, device, batch_size, seed=None):
    X, y = [], []
    for i in tqdm(range(0, len(dataset), batch_size), desc="extracting embeddings"):
        batch = dataset[i:i + batch_size]
        waveforms = [load_waveform({"audio": a}, seed=seed) for a in batch["audio"]]
        emb = embed_batch(waveforms, extractor, model, layer, device)
        X.append(emb)
        y.extend(label2id[label] for label in batch[label_column])
    return np.concatenate(X, axis=0), np.array(y)


def load_or_compute_embeddings(
    dataset,
    label_column,
    label2id,
    model_name,
    layer,
    device,
    batch_size,
    seed,
    cache_dir=None,
):
    cache_path = get_embeddings_cache_path(dataset, label_column, model_name, layer, device, seed, cache_dir)
    cached = load_cached_embeddings(cache_path)
    if cached is not None:
        print(f"loaded cached embeddings from {cache_path}")
        return cached

    extractor, model = load_wavlm(model_name, device)
    X, y = compute_embeddings(dataset, label_column, label2id, extractor, model, layer, device, batch_size, seed=seed)
    save_cached_embeddings(cache_path, X, y)
    print(f"saved embeddings cache to {cache_path}")
    return X, y


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


def train_eval(X_train, y_train, X_test, y_test, id2label, args):
    device = args.device
    class_ids, class_counts = np.unique(y_train, return_counts=True)
    distribution = ", ".join(
        f"{id2label[int(class_id)]}: {int(class_count)}" for class_id, class_count in zip(class_ids, class_counts)
    )
    print(f"train class distribution: {distribution}")

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = safe_stratified_split(X_train, y_train, args.val_ratio, args.seed)
    if len(X_val) == 0:
        raise ValueError(
            f"Validation split is empty for val_ratio={args.val_ratio}. Increase the training set or validation ratio."
        )

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=args.clf_batch_size, shuffle=shuffle)

    train_loader = to_loader(X_tr, y_tr, True)
    val_loader = to_loader(X_val, y_val, False)
    test_loader = to_loader(X_test, y_test, False)

    print(f"split sizes -> train: {len(X_tr)} | val: {len(X_val)} | test: {len(X_test)}")

    num_classes = len(id2label)
    model = ResidualMLP(X_train.shape[1], num_classes, args.hidden_dim, args.num_blocks, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.clf_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def compute_loss(logits, targets):
        if args.loss == "focal":
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            return ((1.0 - pt) ** args.focal_gamma * ce_loss).mean()
        return F.cross_entropy(logits, targets)

    best_val_acc, best_state, patience_left = 0.0, None, args.patience
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
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
        else:
            patience_left -= 1
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
    return acc, report


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
