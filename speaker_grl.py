import copy
import csv
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from wavlm_common import (
    ResidualMLP,
    SAMPLE_RATE,
    build_arg_parser,
    build_label_mapping,
    filter_labeled,
    filter_to_labels,
    get_label_list,
    get_metrics_csv_path,
    load_split,
    load_waveform,
    set_seed,
)
from accent_label_mapping import normalize_accent_labels

SPEAKER_COLUMN = "client_id"
GENDER_COLUMN = "gender"
ACCENT_COLUMN = "accent"
RAW_AGE_COLUMN = "age"
AGE_COLUMN = "age_bin"

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
    "epoch_time_s",
    "eta_s",
    "best_epoch",
    "best_val_speaker_acc",
    "test_speaker_acc",
    "test_age_acc",
    "test_gender_acc",
    "test_accent_acc",
    "speaker_classes",
    "age_classes",
    "gender_classes",
    "accent_classes",
    "model",
    "layer",
    "trainable_layers",
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
    parser.add_argument(
        "--trainable_layers",
        type=int,
        default=4,
        help="number of top WavLM transformer layers to fine-tune; use -1 to freeze the encoder",
    )
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


def filter_to_task_labels(dataset, speaker_labels, age_labels, gender_labels, accent_labels):
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


def load_wavlm_encoder(model_name, device, trainable_layers):
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if trainable_layers != -1 and trainable_layers > 0:
        for block in model.encoder.layers[-trainable_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.encoder.layer_norm.parameters():
            parameter.requires_grad = True
    model.to(device)
    return extractor, model


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
    def __init__(
        self,
        encoder,
        layer,
        speaker_classes,
        age_classes,
        gender_classes,
        accent_classes,
        args,
    ):
        super().__init__()
        self.encoder = encoder
        self.layer = layer
        self.shared_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.encoder.config.hidden_size),
            torch.nn.Linear(self.encoder.config.hidden_size, args.shared_dim),
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

    def mean_pool(self, input_values, attention_mask):
        outputs = self.encoder(input_values, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[self.layer]
        feature_mask = self.encoder._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        feature_mask = feature_mask.unsqueeze(-1).float()
        pooled = (hidden * feature_mask).sum(1) / feature_mask.sum(1).clamp(min=1)
        return self.shared_projection(pooled)

    def forward(
        self,
        input_values,
        attention_mask,
        speaker_labels=None,
        age_labels=None,
        gender_labels=None,
        accent_labels=None,
        grl_lambda=1.0,
    ):
        shared_embedding = self.mean_pool(input_values, attention_mask)
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


def make_collate_fn(speaker2id, age2id, gender2id, accent2id):
    def collate(batch):
        waveforms = [load_waveform(item["audio"]) for item in batch]
        return {
            "waveforms": waveforms,
            "speaker_labels": torch.tensor([speaker2id[item[SPEAKER_COLUMN]] for item in batch], dtype=torch.long),
            "age_labels": torch.tensor([age2id[item[AGE_COLUMN]] for item in batch], dtype=torch.long),
            "gender_labels": torch.tensor([gender2id[item[GENDER_COLUMN]] for item in batch], dtype=torch.long),
            "accent_labels": torch.tensor([accent2id[item[ACCENT_COLUMN]] for item in batch], dtype=torch.long),
        }

    return collate


def batch_to_model_inputs(batch, extractor, device):
    audio_inputs = extractor(
        [waveform.numpy() for waveform in batch["waveforms"]],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = audio_inputs.input_values.to(device)
    attention_mask = audio_inputs.attention_mask.to(device)
    labels = {
        "speaker_labels": batch["speaker_labels"].to(device),
        "age_labels": batch["age_labels"].to(device),
        "gender_labels": batch["gender_labels"].to(device),
        "accent_labels": batch["accent_labels"].to(device),
    }
    return input_values, attention_mask, labels


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
    total_loss = losses["speaker_loss"]
    total_loss = total_loss + args.alpha * losses["age_loss"]
    total_loss = total_loss + args.beta * losses["gender_loss"]
    total_loss = total_loss + args.gamma * losses["accent_loss"]
    return total_loss, losses


def accuracy_from_logits(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).sum().item(), labels.size(0)


def run_split(model, loader, extractor, device, args, train=False, optimizer=None, step_offset=0):
    total_loss_sum = 0.0
    speaker_loss_sum = 0.0
    age_loss_sum = 0.0
    gender_loss_sum = 0.0
    accent_loss_sum = 0.0
    speaker_correct = 0
    age_correct = 0
    gender_correct = 0
    accent_correct = 0
    total_examples = 0

    if train:
        model.train()
    else:
        model.eval()

    for step, batch in enumerate(loader, start=step_offset):
        input_values, attention_mask, labels = batch_to_model_inputs(batch, extractor, device)
        grl_lambda = ramp_lambda(step, args.grl_ramp_steps)

        with torch.set_grad_enabled(train):
            outputs = model(
                input_values,
                attention_mask,
                speaker_labels=labels["speaker_labels"],
                age_labels=labels["age_labels"],
                gender_labels=labels["gender_labels"],
                accent_labels=labels["accent_labels"],
                grl_lambda=grl_lambda,
            )
            total_loss, losses = compute_totals(outputs, args)

        batch_size = labels["speaker_labels"].size(0)
        total_loss_sum += total_loss.item() * batch_size
        speaker_loss_sum += losses["speaker_loss"].item() * batch_size
        age_loss_sum += losses["age_loss"].item() * batch_size
        gender_loss_sum += losses["gender_loss"].item() * batch_size
        accent_loss_sum += losses["accent_loss"].item() * batch_size

        speaker_hits, speaker_total = accuracy_from_logits(outputs["speaker_logits"], labels["speaker_labels"])
        age_hits, age_total = accuracy_from_logits(outputs["age_logits"], labels["age_labels"])
        gender_hits, gender_total = accuracy_from_logits(outputs["gender_logits"], labels["gender_labels"])
        accent_hits, accent_total = accuracy_from_logits(outputs["accent_logits"], labels["accent_labels"])
        speaker_correct += speaker_hits
        age_correct += age_hits
        gender_correct += gender_hits
        accent_correct += accent_hits
        total_examples += speaker_total

        if train:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return {
        "total_loss": total_loss_sum / max(total_examples, 1),
        "speaker_loss": speaker_loss_sum / max(total_examples, 1),
        "age_loss": age_loss_sum / max(total_examples, 1),
        "gender_loss": gender_loss_sum / max(total_examples, 1),
        "accent_loss": accent_loss_sum / max(total_examples, 1),
        "speaker_acc": speaker_correct / max(total_examples, 1),
        "age_acc": age_correct / max(total_examples, 1),
        "gender_acc": gender_correct / max(total_examples, 1),
        "accent_acc": accent_correct / max(total_examples, 1),
        "total_examples": total_examples,
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


def evaluate_reports(model, loader, extractor, device, age2id, id2age, gender2id, id2gender, accent2id, id2accent):
    model.eval()
    speaker_preds = []
    speaker_targets = []
    age_preds = []
    age_targets = []
    gender_preds = []
    gender_targets = []
    accent_preds = []
    accent_targets = []

    with torch.no_grad():
        for batch in loader:
            input_values, attention_mask, labels = batch_to_model_inputs(batch, extractor, device)
            outputs = model(
                input_values,
                attention_mask,
                speaker_labels=None,
                age_labels=None,
                gender_labels=None,
                accent_labels=None,
                grl_lambda=1.0,
            )
            speaker_preds.extend(outputs["speaker_logits"].argmax(dim=1).cpu().tolist())
            speaker_targets.extend(labels["speaker_labels"].tolist())
            age_preds.extend(outputs["age_logits"].argmax(dim=1).cpu().tolist())
            age_targets.extend(labels["age_labels"].tolist())
            gender_preds.extend(outputs["gender_logits"].argmax(dim=1).cpu().tolist())
            gender_targets.extend(labels["gender_labels"].tolist())
            accent_preds.extend(outputs["accent_logits"].argmax(dim=1).cpu().tolist())
            accent_targets.extend(labels["accent_labels"].tolist())

    speaker_acc = accuracy_score(speaker_targets, speaker_preds)
    age_acc = accuracy_score(age_targets, age_preds)
    gender_acc = accuracy_score(gender_targets, gender_preds)
    accent_acc = accuracy_score(accent_targets, accent_preds)

    reports = {
        "age": report_for_labels(age_targets, age_preds, id2age),
        "gender": report_for_labels(gender_targets, gender_preds, id2gender),
        "accent": report_for_labels(accent_targets, accent_preds, id2accent),
    }

    return {
        "speaker_acc": speaker_acc,
        "age_acc": age_acc,
        "gender_acc": gender_acc,
        "accent_acc": accent_acc,
        "reports": reports,
    }


def main():
    parser = build_arg_parser("WavLM speaker GRL training on Common Voice 17 English")
    add_grl_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    train_dataset = prepare_split(args.train_split, args.max_train_samples, args.dataset_repo)
    val_dataset = prepare_split(args.val_split, args.max_val_samples, args.dataset_repo)
    test_dataset = prepare_split(args.test_split, args.max_test_samples, args.dataset_repo)

    if len(train_dataset) == 0:
        raise ValueError("No fully labeled training examples were found for speaker GRL training.")

    speaker_labels = get_label_list(train_dataset, SPEAKER_COLUMN)
    gender_labels = get_label_list(train_dataset, GENDER_COLUMN)
    accent_labels = get_label_list(train_dataset, ACCENT_COLUMN)
    age_labels = [label for label in ["teens", "twenties", "thirties", "fourties", "fifties_plus"] if label in set(train_dataset[AGE_COLUMN])]
    if not age_labels:
        raise ValueError("No age-bin labels were found in the training split after binning.")

    train_dataset = filter_to_task_labels(train_dataset, speaker_labels, age_labels, gender_labels, accent_labels)
    val_dataset = filter_to_task_labels(val_dataset, speaker_labels, age_labels, gender_labels, accent_labels)
    test_dataset = filter_to_task_labels(test_dataset, speaker_labels, age_labels, gender_labels, accent_labels)

    if len(val_dataset) == 0:
        raise ValueError("No labeled validation examples remain after filtering to training labels.")
    if len(test_dataset) == 0:
        raise ValueError("No labeled evaluation examples remain after filtering to training labels.")

    speaker2id, id2speaker = build_label_mapping(speaker_labels)
    age2id, id2age = build_label_mapping(age_labels)
    gender2id, id2gender = build_label_mapping(gender_labels)
    accent2id, id2accent = build_label_mapping(accent_labels)

    print(f"speaker classes ({len(speaker2id)}): {len(speaker2id)} unique client_id values")
    print(f"age classes ({len(age2id)}): {age_labels}")
    print(f"gender classes ({len(gender2id)}): {gender_labels}")
    print(f"accent classes ({len(accent2id)}): {accent_labels}")
    print(f"train samples: {len(train_dataset)} | val samples: {len(val_dataset)} | test samples: {len(test_dataset)}")

    train_age_counts = Counter(train_dataset[AGE_COLUMN])
    train_gender_counts = Counter(train_dataset[GENDER_COLUMN])
    train_accent_counts = Counter(train_dataset[ACCENT_COLUMN])
    print("train age distribution: " + ", ".join(f"{label}: {train_age_counts.get(label, 0)}" for label in age_labels))
    print(
        "train gender distribution: "
        + ", ".join(f"{label}: {train_gender_counts.get(label, 0)}" for label in gender_labels)
    )
    print(
        "train accent distribution: "
        + ", ".join(f"{label}: {train_accent_counts.get(label, 0)}" for label in accent_labels)
    )

    extractor, encoder = load_wavlm_encoder(args.model, args.device, args.trainable_layers)
    model = SpeakerGRLModel(
        encoder,
        args.layer,
        len(speaker2id),
        len(age2id),
        len(gender2id),
        len(accent2id),
        args,
    ).to(args.device)

    collate_fn = make_collate_fn(speaker2id, age2id, gender2id, accent2id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.clf_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_id = time.strftime("%Y%m%d-%H%M%S")
    best_val_speaker_acc = -1.0
    best_epoch = None
    best_state = None
    patience_left = args.patience
    epoch_rows = []
    start_time = time.perf_counter()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_split(
            model,
            train_loader,
            extractor,
            args.device,
            args,
            train=True,
            optimizer=optimizer,
            step_offset=global_step,
        )
        global_step += len(train_loader)
        scheduler.step()

        val_metrics = run_split(model, val_loader, extractor, args.device, args, train=False)
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_elapsed = (time.perf_counter() - start_time) / (epoch + 1)
        remaining_seconds = avg_elapsed * (args.epochs - epoch - 1)
        eta_minutes, eta_seconds = divmod(int(round(remaining_seconds)), 60)

        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"speaker loss {train_metrics['speaker_loss']:.4f} | total loss {train_metrics['total_loss']:.4f} | "
            f"val speaker acc {val_metrics['speaker_acc']:.4f} | val age acc {val_metrics['age_acc']:.4f} | "
            f"val gender acc {val_metrics['gender_acc']:.4f} | val accent acc {val_metrics['accent_acc']:.4f} | "
            f"epoch time {epoch_elapsed:.1f}s | eta {eta_minutes:02d}:{eta_seconds:02d}"
        )

        if val_metrics["speaker_acc"] > best_val_speaker_acc:
            best_val_speaker_acc = val_metrics["speaker_acc"]
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
                "epoch_time_s": epoch_elapsed,
                "eta_s": remaining_seconds,
                "best_epoch": best_epoch,
                "best_val_speaker_acc": best_val_speaker_acc,
                "test_speaker_acc": None,
                "test_age_acc": None,
                "test_gender_acc": None,
                "test_accent_acc": None,
                "speaker_classes": len(speaker2id),
                "age_classes": len(age2id),
                "gender_classes": len(gender2id),
                "accent_classes": len(accent2id),
                "model": args.model,
                "layer": args.layer,
                "trainable_layers": args.trainable_layers,
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

    test_metrics = run_split(model, test_loader, extractor, args.device, args, train=False)
    test_reports = evaluate_reports(model, test_loader, extractor, args.device, age2id, id2age, gender2id, id2gender, accent2id, id2accent)

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
            "epoch_time_s": None,
            "eta_s": None,
            "best_epoch": best_epoch,
            "best_val_speaker_acc": best_val_speaker_acc,
            "test_speaker_acc": test_metrics["speaker_acc"],
            "test_age_acc": test_metrics["age_acc"],
            "test_gender_acc": test_metrics["gender_acc"],
            "test_accent_acc": test_metrics["accent_acc"],
            "speaker_classes": len(speaker2id),
            "age_classes": len(age2id),
            "gender_classes": len(gender2id),
            "accent_classes": len(accent2id),
            "model": args.model,
            "layer": args.layer,
            "trainable_layers": args.trainable_layers,
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

    print(f"best val speaker acc: {best_val_speaker_acc:.4f} at epoch {best_epoch}")
    print(f"test speaker acc: {test_metrics['speaker_acc']:.4f}")
    print(f"test age acc: {test_metrics['age_acc']:.4f}")
    print(f"test gender acc: {test_metrics['gender_acc']:.4f}")
    print(f"test accent acc: {test_metrics['accent_acc']:.4f}")
    print(test_reports["reports"]["age"])
    print(test_reports["reports"]["gender"])
    print(test_reports["reports"]["accent"])
    print(f"speaker GRL metrics csv: {metrics_path}")


if __name__ == "__main__":
    main()