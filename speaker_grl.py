import copy
import csv
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from wavlm_common import (
    ResidualMLP,
    build_arg_parser,
    build_label_mapping,
    filter_labeled,
    filter_to_labels,
    get_label_list,
    get_metrics_csv_path,
    load_embeddings,
    load_split,
    sample_verification_trials,
    set_seed,
    verification_metrics,
)
from accent_label_mapping import normalize_accent_labels

SPEAKER_COLUMN = "client_id"
GENDER_COLUMN = "gender"
ACCENT_COLUMN = "accent"
RAW_AGE_COLUMN = "age"
AGE_COLUMN = "age_bin"

TRAIN_CHUNKS = 3
EVAL_CHUNKS = 1

AGE_BINS = {
    "teens": "teens",
    "twenties": "twenties",
    "thirties": "thirties",
    "fourties": "fourties",
    "fifties": "fifties_plus",
    "sixties": "fifties_plus",
    "seventies": "fifties_plus",
    "eighties": "fifties_plus",
    "nineties": "fifties_plus",
}


MULTITASK_METRICS_FIELDNAMES = [
    "run_id",
    "task",
    "stage",
    "epoch",
    "total_loss",
    "speaker_loss",
    "age_loss",
    "gender_loss",
    "accent_loss",
    "speaker_acc",
    "age_acc",
    "gender_acc",
    "accent_acc",
    "speaker_eer",
    "speaker_auc",
    "epoch_time_s",
    "eta_s",
    "best_epoch",
    "best_val_speaker_eer",
    "test_speaker_eer",
    "test_speaker_auc",
    "test_age_acc",
    "test_gender_acc",
    "test_accent_acc",
    "speaker_classes",
    "age_classes",
    "gender_classes",
    "accent_classes",
    "model",
    "layer",
    "seed",
    "alpha",
    "beta",
    "gamma",
    "speaker_margin",
    "speaker_scale",
    "grl_ramp_steps",
    "device",
]


def add_grl_args(parser):
    parser.add_argument("--shared_dim", type=int, default=512, help="dimension of the pooled shared embedding")
    parser.add_argument(
        "--speaker_embed_dim",
        type=int,
        default=256,
        help="dimension of the speaker embedding before AAM-Softmax",
    )
    parser.add_argument(
        "--speaker_margin",
        type=float,
        default=0.2,
        help="additive angular margin used by the speaker classifier",
    )
    parser.add_argument(
        "--speaker_scale",
        type=float,
        default=30.0,
        help="logit scale used by the speaker classifier",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for the age adversarial loss")
    parser.add_argument("--beta", type=float, default=1.0, help="weight for the gender adversarial loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="weight for the accent adversarial loss")
    parser.add_argument(
        "--grl_ramp_steps",
        type=int,
        default=1000,
        help="number of optimization steps used to ramp the GRL lambda from 0 to 1",
    )
    return parser


def add_age_bin(example):
    return {AGE_COLUMN: AGE_BINS.get(example[RAW_AGE_COLUMN])}


def prepare_split(split, max_samples, dataset_repo):
    dataset = load_split(split, max_samples, dataset_repo)
    dataset = filter_labeled(dataset, SPEAKER_COLUMN)
    dataset = filter_labeled(dataset, GENDER_COLUMN)
    dataset = normalize_accent_labels(dataset, ACCENT_COLUMN)
    dataset = filter_labeled(dataset, ACCENT_COLUMN)
    dataset = filter_labeled(dataset, RAW_AGE_COLUMN)
    dataset = dataset.map(add_age_bin)
    dataset = filter_labeled(dataset, AGE_COLUMN)
    return dataset


def filter_to_task_labels(dataset, age_labels, gender_labels, accent_labels, speaker_labels=None):
    if speaker_labels is not None:
        dataset = filter_to_labels(dataset, SPEAKER_COLUMN, speaker_labels)
    dataset = filter_to_labels(dataset, AGE_COLUMN, age_labels)
    dataset = filter_to_labels(dataset, GENDER_COLUMN, gender_labels)
    dataset = filter_to_labels(dataset, ACCENT_COLUMN, accent_labels)
    return dataset


def save_metrics_csv(metrics_path, rows):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MULTITASK_METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def labels_to_ids(dataset, column, label2id):
    return np.array([label2id[label] for label in dataset[column]], dtype=np.int64)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, lambd):
        ctx.lambd = lambd
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(torch.nn.Module):
    def forward(self, inputs, lambd):
        return GradientReversalFunction.apply(inputs, lambd)


class AAMSoftmaxHead(torch.nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = torch.nn.Parameter(torch.empty(num_classes, embedding_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def logits(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(embeddings, weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        threshold = math.cos(math.pi - self.margin)
        margin_correction = math.sin(math.pi - self.margin) * self.margin
        phi = torch.where(cosine > threshold, phi, cosine - margin_correction)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)
        loss = F.cross_entropy(logits, labels)
        return logits, loss


class AdversarialHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(inputs)


class SpeakerGRLModel(torch.nn.Module):
    """Operates on precomputed (frozen) WavLM pooled embeddings, not raw audio.

    The WavLM encoder is never part of this module or its gradient graph;
    embeddings are precomputed once by wavlm_common.load_embeddings and cached
    to disk, so this model only needs the embedding dimensionality.
    """

    def __init__(
        self,
        embedding_dim,
        speaker_classes,
        age_classes,
        gender_classes,
        accent_classes,
        args,
    ):
        super().__init__()
        self.shared_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, args.shared_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.speaker_projection = ResidualMLP(
            args.shared_dim,
            args.speaker_embed_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
        )
        self.speaker_head = AAMSoftmaxHead(
            args.speaker_embed_dim,
            speaker_classes,
            margin=args.speaker_margin,
            scale=args.speaker_scale,
        )
        self.grl = GradientReversal()
        self.age_head = AdversarialHead(args.shared_dim, age_classes, args.dropout)
        self.gender_head = AdversarialHead(args.shared_dim, gender_classes, args.dropout)
        self.accent_head = AdversarialHead(args.shared_dim, accent_classes, args.dropout)

    def forward(
        self,
        features,
        speaker_labels=None,
        age_labels=None,
        gender_labels=None,
        accent_labels=None,
        grl_lambda=1.0,
    ):
        shared_embedding = self.shared_projection(features)
        speaker_embedding = self.speaker_projection(shared_embedding)
        speaker_logits = self.speaker_head.logits(speaker_embedding)

        speaker_loss = None
        if speaker_labels is not None:
            _, speaker_loss = self.speaker_head(speaker_embedding, speaker_labels)

        reversed_embedding = self.grl(shared_embedding, grl_lambda)
        age_logits = self.age_head(reversed_embedding)
        gender_logits = self.gender_head(reversed_embedding)
        accent_logits = self.accent_head(reversed_embedding)

        age_loss = F.cross_entropy(age_logits, age_labels) if age_labels is not None else None
        gender_loss = F.cross_entropy(gender_logits, gender_labels) if gender_labels is not None else None
        accent_loss = F.cross_entropy(accent_logits, accent_labels) if accent_labels is not None else None

        return {
            "shared_embedding": shared_embedding,
            "speaker_embedding": speaker_embedding,
            "speaker_logits": speaker_logits,
            "speaker_loss": speaker_loss,
            "age_logits": age_logits,
            "age_loss": age_loss,
            "gender_logits": gender_logits,
            "gender_loss": gender_loss,
            "accent_logits": accent_logits,
            "accent_loss": accent_loss,
        }


def ramp_lambda(step, ramp_steps):
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, step / float(ramp_steps))


def compute_totals(outputs, args):
    losses = {
        "speaker_loss": outputs["speaker_loss"],
        "age_loss": outputs["age_loss"],
        "gender_loss": outputs["gender_loss"],
        "accent_loss": outputs["accent_loss"],
    }
    total_loss = (
        args.alpha * losses["age_loss"]
        + args.beta * losses["gender_loss"]
        + args.gamma * losses["accent_loss"]
    )
    if losses["speaker_loss"] is not None:
        total_loss = total_loss + losses["speaker_loss"]
    return total_loss, losses


def accuracy_from_logits(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).sum().item(), labels.size(0)


def iter_batches(n, batch_size):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def run_split(
    model,
    features,
    device,
    args,
    age_labels,
    gender_labels,
    accent_labels,
    speaker_labels=None,
    train=False,
    optimizer=None,
    step_offset=0,
    shuffle_generator=None,
):
    """features: np.ndarray [num_samples, hidden_dim] (already scaled)."""
    num_samples = features.shape[0]
    order = np.arange(num_samples)
    if shuffle_generator is not None:
        shuffle_generator.shuffle(order)

    total_loss_sum = 0.0
    speaker_loss_sum = 0.0
    age_loss_sum = 0.0
    gender_loss_sum = 0.0
    accent_loss_sum = 0.0
    speaker_correct = 0
    speaker_examples = 0
    age_correct = 0
    gender_correct = 0
    accent_correct = 0
    total_examples = 0

    if train:
        model.train()
    else:
        model.eval()

    features_t_full = torch.tensor(features, dtype=torch.float32)
    age_t_full = torch.tensor(age_labels, dtype=torch.long)
    gender_t_full = torch.tensor(gender_labels, dtype=torch.long)
    accent_t_full = torch.tensor(accent_labels, dtype=torch.long)
    speaker_t_full = torch.tensor(speaker_labels, dtype=torch.long) if speaker_labels is not None else None

    for step, (start, end) in enumerate(iter_batches(num_samples, args.clf_batch_size), start=step_offset):
        idx = order[start:end]
        feats = features_t_full[idx].to(device)
        age_b = age_t_full[idx].to(device)
        gender_b = gender_t_full[idx].to(device)
        accent_b = accent_t_full[idx].to(device)
        speaker_b = speaker_t_full[idx].to(device) if speaker_t_full is not None else None

        grl_lambda = ramp_lambda(step, args.grl_ramp_steps)

        with torch.set_grad_enabled(train):
            outputs = model(
                feats,
                speaker_labels=speaker_b,
                age_labels=age_b,
                gender_labels=gender_b,
                accent_labels=accent_b,
                grl_lambda=grl_lambda,
            )
            total_loss, losses = compute_totals(outputs, args)

        batch_size = age_b.size(0)
        total_loss_sum += total_loss.item() * batch_size
        if speaker_b is not None:
            speaker_loss_sum += losses["speaker_loss"].item() * batch_size
            speaker_hits, speaker_total = accuracy_from_logits(outputs["speaker_logits"], speaker_b)
            speaker_correct += speaker_hits
            speaker_examples += speaker_total
        age_loss_sum += losses["age_loss"].item() * batch_size
        gender_loss_sum += losses["gender_loss"].item() * batch_size
        accent_loss_sum += losses["accent_loss"].item() * batch_size

        age_hits, age_total = accuracy_from_logits(outputs["age_logits"], age_b)
        gender_hits, gender_total = accuracy_from_logits(outputs["gender_logits"], gender_b)
        accent_hits, accent_total = accuracy_from_logits(outputs["accent_logits"], accent_b)
        age_correct += age_hits
        gender_correct += gender_hits
        accent_correct += accent_hits
        total_examples += age_total

        if train:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return {
        "total_loss": total_loss_sum / max(total_examples, 1),
        "speaker_loss": speaker_loss_sum / max(speaker_examples, 1),
        "age_loss": age_loss_sum / max(total_examples, 1),
        "gender_loss": gender_loss_sum / max(total_examples, 1),
        "accent_loss": accent_loss_sum / max(total_examples, 1),
        "speaker_acc": speaker_correct / max(speaker_examples, 1),
        "age_acc": age_correct / max(total_examples, 1),
        "gender_acc": gender_correct / max(total_examples, 1),
        "accent_acc": accent_correct / max(total_examples, 1),
        "total_examples": total_examples,
        "num_batches": (num_samples + args.clf_batch_size - 1) // args.clf_batch_size,
    }


def report_for_labels(targets, predictions, id2label):
    label_ids = sorted(id2label)
    return classification_report(
        targets,
        predictions,
        labels=label_ids,
        target_names=[id2label[label_id] for label_id in label_ids],
        zero_division=0,
    )


@torch.no_grad()
def evaluate_reports(model, features, device, age_labels, gender_labels, accent_labels, args, id2age, id2gender, id2accent):
    model.eval()
    age_preds, gender_preds, accent_preds = [], [], []
    features_t_full = torch.tensor(features, dtype=torch.float32)

    for start, end in iter_batches(features.shape[0], args.clf_batch_size):
        feats = features_t_full[start:end].to(device)
        outputs = model(feats, grl_lambda=1.0)
        age_preds.extend(outputs["age_logits"].argmax(dim=1).cpu().tolist())
        gender_preds.extend(outputs["gender_logits"].argmax(dim=1).cpu().tolist())
        accent_preds.extend(outputs["accent_logits"].argmax(dim=1).cpu().tolist())

    age_acc = accuracy_score(age_labels, age_preds)
    gender_acc = accuracy_score(gender_labels, gender_preds)
    accent_acc = accuracy_score(accent_labels, accent_preds)

    reports = {
        "age": report_for_labels(age_labels, age_preds, id2age),
        "gender": report_for_labels(gender_labels, gender_preds, id2gender),
        "accent": report_for_labels(accent_labels, accent_preds, id2accent),
    }

    return {
        "age_acc": age_acc,
        "gender_acc": gender_acc,
        "accent_acc": accent_acc,
        "reports": reports,
    }


@torch.no_grad()
def extract_embeddings_by_speaker(model, features, speaker_ids, device, batch_size):
    model.eval()
    embeddings_by_speaker = defaultdict(list)
    features_t_full = torch.tensor(features, dtype=torch.float32)

    for start, end in iter_batches(features.shape[0], batch_size):
        feats = features_t_full[start:end].to(device)
        outputs = model(feats, grl_lambda=1.0)
        embeddings = outputs["speaker_embedding"].cpu().numpy()
        for speaker_id, embedding in zip(speaker_ids[start:end], embeddings):
            embeddings_by_speaker[speaker_id].append(embedding)
    return embeddings_by_speaker


def speaker_verification_metrics(model, features, speaker_ids, device, batch_size, seed):
    embeddings_by_speaker = extract_embeddings_by_speaker(model, features, speaker_ids, device, batch_size)
    left, right, labels = sample_verification_trials(embeddings_by_speaker, seed=seed)
    return verification_metrics(left, right, labels)


def main():
    parser = build_arg_parser("WavLM speaker GRL training on Common Voice 17 English (frozen WavLM features)")
    add_grl_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    train_dataset = prepare_split(args.train_split, args.max_train_samples, args.dataset_repo)
    val_dataset = prepare_split(args.val_split, args.max_val_samples, args.dataset_repo)
    test_dataset = prepare_split(args.test_split, args.max_test_samples, args.dataset_repo)

    if len(train_dataset) == 0:
        raise ValueError("No fully labeled training examples were found for speaker GRL training.")

    speaker_labels_vocab = get_label_list(train_dataset, SPEAKER_COLUMN)
    gender_labels_vocab = get_label_list(train_dataset, GENDER_COLUMN)
    accent_labels_vocab = get_label_list(train_dataset, ACCENT_COLUMN)
    age_labels_vocab = [label for label in ["teens", "twenties", "thirties", "fourties", "fifties_plus"] if label in set(train_dataset[AGE_COLUMN])]
    if not age_labels_vocab:
        raise ValueError("No age-bin labels were found in the training split after binning.")

    # Speaker ID is evaluated open-set (verification), so val/test are only
    # filtered to the shared age/gender/accent label vocab, not to train speakers.
    train_dataset = filter_to_task_labels(train_dataset, age_labels_vocab, gender_labels_vocab, accent_labels_vocab, speaker_labels=speaker_labels_vocab)
    val_dataset = filter_to_task_labels(val_dataset, age_labels_vocab, gender_labels_vocab, accent_labels_vocab)
    test_dataset = filter_to_task_labels(test_dataset, age_labels_vocab, gender_labels_vocab, accent_labels_vocab)

    if len(val_dataset) == 0:
        raise ValueError("No labeled validation examples remain after filtering to training labels.")
    if len(test_dataset) == 0:
        raise ValueError("No labeled evaluation examples remain after filtering to training labels.")

    speaker2id, id2speaker = build_label_mapping(speaker_labels_vocab)
    age2id, id2age = build_label_mapping(age_labels_vocab)
    gender2id, id2gender = build_label_mapping(gender_labels_vocab)
    accent2id, id2accent = build_label_mapping(accent_labels_vocab)

    print(f"speaker classes ({len(speaker2id)}): {len(speaker2id)} unique client_id values")
    print(f"age classes ({len(age2id)}): {age_labels_vocab}")
    print(f"gender classes ({len(gender2id)}): {gender_labels_vocab}")
    print(f"accent classes ({len(accent2id)}): {accent_labels_vocab}")
    print(f"train samples: {len(train_dataset)} | val samples: {len(val_dataset)} | test samples: {len(test_dataset)}")

    train_age_counts = Counter(train_dataset[AGE_COLUMN])
    train_gender_counts = Counter(train_dataset[GENDER_COLUMN])
    train_accent_counts = Counter(train_dataset[ACCENT_COLUMN])
    print("train age distribution: " + ", ".join(f"{label}: {train_age_counts.get(label, 0)}" for label in age_labels_vocab))
    print(
        "train gender distribution: "
        + ", ".join(f"{label}: {train_gender_counts.get(label, 0)}" for label in gender_labels_vocab)
    )
    print(
        "train accent distribution: "
        + ", ".join(f"{label}: {train_accent_counts.get(label, 0)}" for label in accent_labels_vocab)
    )

    # --- Precompute (or load cached) frozen WavLM embeddings ---
    # WavLM is never fine-tuned: features are extracted once per split, cached
    # to disk, and reused across epochs (with `TRAIN_CHUNKS` crop variations
    # for the training split to keep some augmentation without touching the encoder).
    print(f"loading/precomputing frozen WavLM embeddings from {args.model} (layer {args.layer})...")
    X_train, y_train_speaker = load_embeddings(
        train_dataset,
        split_name="train",
        label_column=SPEAKER_COLUMN,
        label2id=speaker2id,
        model_name=args.model,
        layer=args.layer,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        num_chunks=TRAIN_CHUNKS,
    )
    X_val, y_val_age = load_embeddings(
        val_dataset,
        split_name="validation",
        label_column=AGE_COLUMN,
        label2id=age2id,
        model_name=args.model,
        layer=args.layer,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        num_chunks=EVAL_CHUNKS,
    )
    X_test, y_test_age = load_embeddings(
        test_dataset,
        split_name="test",
        label_column=AGE_COLUMN,
        label2id=age2id,
        model_name=args.model,
        layer=args.layer,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        num_chunks=EVAL_CHUNKS,
    )

    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=0)
    if X_val.ndim == 2:
        X_val = np.expand_dims(X_val, axis=0)
    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, axis=0)

    y_train_age = labels_to_ids(train_dataset, AGE_COLUMN, age2id)
    y_train_gender = labels_to_ids(train_dataset, GENDER_COLUMN, gender2id)
    y_train_accent = labels_to_ids(train_dataset, ACCENT_COLUMN, accent2id)

    y_val_gender = labels_to_ids(val_dataset, GENDER_COLUMN, gender2id)
    y_val_accent = labels_to_ids(val_dataset, ACCENT_COLUMN, accent2id)
    val_speaker_ids = val_dataset[SPEAKER_COLUMN]

    y_test_gender = labels_to_ids(test_dataset, GENDER_COLUMN, gender2id)
    y_test_accent = labels_to_ids(test_dataset, ACCENT_COLUMN, accent2id)
    test_speaker_ids = test_dataset[SPEAKER_COLUMN]

    scaler = StandardScaler().fit(X_train[0])
    X_val_scaled = scaler.transform(X_val[0])
    X_test_scaled = scaler.transform(X_test[0])

    embedding_dim = X_train.shape[-1]
    print(f"building speaker GRL model with shared dim {args.shared_dim} and speaker embed dim {args.speaker_embed_dim}...")
    model = SpeakerGRLModel(
        embedding_dim,
        len(speaker2id),
        len(age2id),
        len(gender2id),
        len(accent2id),
        args,
    ).to(args.device)

    print(f"model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters (WavLM is frozen and not part of this model)")

    print(f"training for {args.epochs} epochs with batch size {args.clf_batch_size}, learning rate {args.clf_lr}, weight decay {args.weight_decay}, and patience {args.patience}...")
    print(f"adversarial loss weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.clf_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_id = time.strftime("%Y%m%d-%H%M%S")
    best_val_speaker_eer = None
    best_epoch = None
    best_state = None
    patience_left = args.patience
    epoch_rows = []
    start_time = time.perf_counter()
    global_step = 0
    device = args.device

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()

        # Cycle through precomputed crop variations of the training features.
        ep_idx = epoch % X_train.shape[0]
        X_train_ep = scaler.transform(X_train[ep_idx])
        shuffle_rng = np.random.RandomState(args.seed + epoch)

        train_metrics = run_split(
            model,
            X_train_ep,
            device,
            args,
            age_labels=y_train_age,
            gender_labels=y_train_gender,
            accent_labels=y_train_accent,
            speaker_labels=y_train_speaker,
            train=True,
            optimizer=optimizer,
            step_offset=global_step,
            shuffle_generator=shuffle_rng,
        )
        global_step += train_metrics["num_batches"]
        scheduler.step()

        val_metrics = run_split(
            model,
            X_val_scaled,
            device,
            args,
            age_labels=y_val_age,
            gender_labels=y_val_gender,
            accent_labels=y_val_accent,
            speaker_labels=None,
            train=False,
        )
        val_verification = speaker_verification_metrics(model, X_val_scaled, val_speaker_ids, device, args.clf_batch_size, args.seed)
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_elapsed = (time.perf_counter() - start_time) / (epoch + 1)
        remaining_seconds = avg_elapsed * (args.epochs - epoch - 1)
        eta_minutes, eta_seconds = divmod(int(round(remaining_seconds)), 60)

        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"speaker loss {train_metrics['speaker_loss']:.4f} | total loss {train_metrics['total_loss']:.4f} | "
            f"val speaker EER {val_verification['eer']:.4f} | val age acc {val_metrics['age_acc']:.4f} | "
            f"val gender acc {val_metrics['gender_acc']:.4f} | val accent acc {val_metrics['accent_acc']:.4f} | "
            f"epoch time {epoch_elapsed:.1f}s | eta {eta_minutes:02d}:{eta_seconds:02d}"
        )

        if best_val_speaker_eer is None or val_verification["eer"] < best_val_speaker_eer:
            best_val_speaker_eer = val_verification["eer"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1

        epoch_rows.append(
            {
                "run_id": run_id,
                "task": "speaker_grl",
                "stage": "epoch",
                "epoch": epoch + 1,
                "total_loss": train_metrics["total_loss"],
                "speaker_loss": train_metrics["speaker_loss"],
                "age_loss": train_metrics["age_loss"],
                "gender_loss": train_metrics["gender_loss"],
                "accent_loss": train_metrics["accent_loss"],
                "speaker_acc": train_metrics["speaker_acc"],
                "age_acc": train_metrics["age_acc"],
                "gender_acc": train_metrics["gender_acc"],
                "accent_acc": train_metrics["accent_acc"],
                "speaker_eer": val_verification["eer"],
                "speaker_auc": val_verification["roc_auc"],
                "epoch_time_s": epoch_elapsed,
                "eta_s": remaining_seconds,
                "best_epoch": best_epoch,
                "best_val_speaker_eer": best_val_speaker_eer,
                "test_speaker_eer": None,
                "test_speaker_auc": None,
                "test_age_acc": None,
                "test_gender_acc": None,
                "test_accent_acc": None,
                "speaker_classes": len(speaker2id),
                "age_classes": len(age2id),
                "gender_classes": len(gender2id),
                "accent_classes": len(accent2id),
                "model": args.model,
                "layer": args.layer,
                "seed": args.seed,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "speaker_margin": args.speaker_margin,
                "speaker_scale": args.speaker_scale,
                "grl_ramp_steps": args.grl_ramp_steps,
                "device": args.device,
            }
        )

        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_split(
        model,
        X_test_scaled,
        device,
        args,
        age_labels=y_test_age,
        gender_labels=y_test_gender,
        accent_labels=y_test_accent,
        speaker_labels=None,
        train=False,
    )
    test_verification = speaker_verification_metrics(model, X_test_scaled, test_speaker_ids, device, args.clf_batch_size, args.seed)
    test_reports = evaluate_reports(
        model, X_test_scaled, device, y_test_age, y_test_gender, y_test_accent, args, id2age, id2gender, id2accent
    )

    metrics_rows = [
        *epoch_rows,
        {
            "run_id": run_id,
            "task": "speaker_grl",
            "stage": "final",
            "epoch": None,
            "total_loss": None,
            "speaker_loss": None,
            "age_loss": None,
            "gender_loss": None,
            "accent_loss": None,
            "speaker_acc": None,
            "age_acc": None,
            "gender_acc": None,
            "accent_acc": None,
            "speaker_eer": None,
            "speaker_auc": None,
            "epoch_time_s": None,
            "eta_s": None,
            "best_epoch": best_epoch,
            "best_val_speaker_eer": best_val_speaker_eer,
            "test_speaker_eer": test_verification["eer"],
            "test_speaker_auc": test_verification["roc_auc"],
            "test_age_acc": test_metrics["age_acc"],
            "test_gender_acc": test_metrics["gender_acc"],
            "test_accent_acc": test_metrics["accent_acc"],
            "speaker_classes": len(speaker2id),
            "age_classes": len(age2id),
            "gender_classes": len(gender2id),
            "accent_classes": len(accent2id),
            "model": args.model,
            "layer": args.layer,
            "seed": args.seed,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "speaker_margin": args.speaker_margin,
            "speaker_scale": args.speaker_scale,
            "grl_ramp_steps": args.grl_ramp_steps,
            "device": args.device,
        },
    ]

    metrics_path = get_metrics_csv_path("speaker_grl", args)
    save_metrics_csv(metrics_path, metrics_rows)

    print(f"best val speaker EER: {best_val_speaker_eer:.4f} at epoch {best_epoch}")
    print(f"test speaker EER: {test_verification['eer']:.4f} | AUC: {test_verification['roc_auc']:.4f}")
    print(f"test age acc: {test_metrics['age_acc']:.4f}")
    print(f"test gender acc: {test_metrics['gender_acc']:.4f}")
    print(f"test accent acc: {test_metrics['accent_acc']:.4f}")
    print(test_reports["reports"]["age"])
    print(test_reports["reports"]["gender"])
    print(test_reports["reports"]["accent"])
    print(f"speaker GRL metrics csv: {metrics_path}")


if __name__ == "__main__":
    main()