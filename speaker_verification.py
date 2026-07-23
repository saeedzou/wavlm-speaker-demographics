import csv
import numpy as np
from collections import Counter
from datasets import concatenate_datasets
from sklearn.model_selection import train_test_split

from wavlm_common import (
    build_arg_parser,
    build_label_mapping,
    filter_labeled,
    get_metrics_csv_path,
    load_embeddings,
    load_split,
    sample_verification_trials,
    set_seed,
    verification_metrics,
)


LABEL_COLUMN = "client_id"

VERIFICATION_METRICS_FIELDNAMES = [
    "run_id",
    "task",
    "stage",
    "roc_auc",
    "eer",
    "threshold",
    "accept_accuracy",
    "reject_rate",
    "speaker_eval_ratio",
    "pos_pairs_per_speaker",
    "neg_pairs_per_speaker",
    "model",
    "layer",
    "seed",
    "device",
]


def save_verification_metrics_csv(metrics_path, rows):
    metrics_path = metrics_path.resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VERIFICATION_METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def embeddings_by_label(embeddings, labels):
    grouped = {}
    for embedding, label in zip(embeddings, labels):
        grouped.setdefault(label, []).append(embedding)
    return grouped


def main():
    parser = build_arg_parser("WavLM open-set speaker verification on Common Voice 17 English")
    parser.add_argument("--speaker_eval_ratio", type=float, default=0.3, help="fraction of speakers reserved for evaluation")
    parser.add_argument("--pos_pairs_per_speaker", type=int, default=10)
    parser.add_argument("--neg_pairs_per_speaker", type=int, default=10)
    args = parser.parse_args()
    set_seed(args.seed)

    train_dataset = filter_labeled(
        load_split(args.train_split, args.max_train_samples, args.dataset_repo), LABEL_COLUMN
    )
    test_dataset = filter_labeled(load_split(args.test_split, args.max_test_samples, args.dataset_repo), LABEL_COLUMN)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Speaker verification requires labeled train and test splits with client_id values.")

    full_dataset = concatenate_datasets([train_dataset, test_dataset])
    label2id, id2label = build_label_mapping(full_dataset[LABEL_COLUMN])
    print(f"speaker identities ({len(label2id)}): {len(label2id)} unique client_id values")
    print(f"combined samples: {len(full_dataset)}")
    # train_counts = Counter(train_dataset[LABEL_COLUMN])
    # test_counts = Counter(test_dataset[LABEL_COLUMN])
    # print(
    #     "train speaker distribution: "
    #     + ", ".join(f"{k}: {v}" for k, v in sorted(train_counts.items(), key=lambda x: (-x[1], x[0])))
    # )
    # print(
    #     "test speaker distribution: "
    #     + ", ".join(f"{k}: {v}" for k, v in sorted(test_counts.items(), key=lambda x: (-x[1], x[0])))
    # )

    embeddings, labels = load_embeddings(
        full_dataset,
        split_name="full",
        label_column=LABEL_COLUMN,
        label2id=label2id,
        model_name=args.model,
        layer=args.layer,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        num_chunks=1,
    )
    # 2. Select the single chunk variation: [1, num_samples, hidden_dim] -> [num_samples, hidden_dim]
    if embeddings.ndim == 3:
        embeddings = embeddings[0]

    unique_speakers = np.unique(labels)
    if unique_speakers.size < 4:
        raise ValueError("Need at least four speaker identities to build disjoint calibration and evaluation sets.")

    calibration_speakers, evaluation_speakers = train_test_split(
        unique_speakers, test_size=args.speaker_eval_ratio, random_state=args.seed
    )
    calibration_mask = np.isin(labels, calibration_speakers)
    evaluation_mask = np.isin(labels, evaluation_speakers)

    calibration_grouped = embeddings_by_label(embeddings[calibration_mask], labels[calibration_mask])
    evaluation_grouped = embeddings_by_label(embeddings[evaluation_mask], labels[evaluation_mask])

    calibration_left, calibration_right, calibration_labels = sample_verification_trials(
        calibration_grouped,
        pos_pairs_per_speaker=args.pos_pairs_per_speaker,
        neg_pairs_per_speaker=args.neg_pairs_per_speaker,
        seed=args.seed,
    )
    calibration_summary = verification_metrics(calibration_left, calibration_right, calibration_labels)

    evaluation_left, evaluation_right, evaluation_labels = sample_verification_trials(
        evaluation_grouped,
        pos_pairs_per_speaker=args.pos_pairs_per_speaker,
        neg_pairs_per_speaker=args.neg_pairs_per_speaker,
        seed=args.seed + 1,
    )
    evaluation_scores = np.sum(evaluation_left * evaluation_right, axis=-1) / (
        np.linalg.norm(evaluation_left, axis=-1) * np.linalg.norm(evaluation_right, axis=-1) + 1e-12
    )
    evaluation_summary = verification_metrics(evaluation_left, evaluation_right, evaluation_labels)
    evaluation_predictions = (evaluation_scores >= calibration_summary["threshold"]).astype(np.int64)
    evaluation_accept_accuracy = float((evaluation_predictions == evaluation_labels).mean())
    evaluation_reject_rate = float(((evaluation_predictions == 0) & (evaluation_labels == 0)).sum() / max((evaluation_labels == 0).sum(), 1))

    metrics_rows = [
        {
            "run_id": f"{args.seed}-{args.model}-{args.layer}",
            "task": "speaker_verification",
            "stage": "calibration",
            "roc_auc": calibration_summary["roc_auc"],
            "eer": calibration_summary["eer"],
            "threshold": calibration_summary["threshold"],
            "accept_accuracy": None,
            "reject_rate": None,
            "speaker_eval_ratio": args.speaker_eval_ratio,
            "pos_pairs_per_speaker": args.pos_pairs_per_speaker,
            "neg_pairs_per_speaker": args.neg_pairs_per_speaker,
            "model": args.model,
            "layer": args.layer,
            "seed": args.seed,
            "device": args.device,
        },
        {
            "run_id": f"{args.seed}-{args.model}-{args.layer}",
            "task": "speaker_verification",
            "stage": "evaluation",
            "roc_auc": evaluation_summary["roc_auc"],
            "eer": evaluation_summary["eer"],
            "threshold": calibration_summary["threshold"],
            "accept_accuracy": evaluation_accept_accuracy,
            "reject_rate": evaluation_reject_rate,
            "speaker_eval_ratio": args.speaker_eval_ratio,
            "pos_pairs_per_speaker": args.pos_pairs_per_speaker,
            "neg_pairs_per_speaker": args.neg_pairs_per_speaker,
            "model": args.model,
            "layer": args.layer,
            "seed": args.seed,
            "device": args.device,
        },
    ]
    metrics_path = get_metrics_csv_path("speaker_verification", args)
    save_verification_metrics_csv(metrics_path, metrics_rows)

    print(f"speaker calibration ROC AUC: {calibration_summary['roc_auc']:.4f}")
    print(f"speaker calibration EER: {calibration_summary['eer']:.4f}")
    print(f"speaker threshold: {calibration_summary['threshold']:.4f}")
    print(f"speaker evaluation ROC AUC: {evaluation_summary['roc_auc']:.4f}")
    print(f"speaker evaluation EER: {evaluation_summary['eer']:.4f}")
    print(f"speaker evaluation accept accuracy: {evaluation_accept_accuracy:.4f}")
    print(f"speaker evaluation reject rate: {evaluation_reject_rate:.4f}")
    print(f"speaker metrics csv: {metrics_path}")


if __name__ == "__main__":
    main()