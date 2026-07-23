import numpy as np
from datasets import concatenate_datasets
from sklearn.model_selection import train_test_split

from wavlm_common import (
    build_arg_parser,
    build_label_mapping,
    filter_labeled,
    load_or_compute_embeddings,
    load_split,
    sample_verification_trials,
    set_seed,
    verification_metrics,
)


LABEL_COLUMN = "client_id"


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

    train_dataset = filter_labeled(load_split(args.train_split, args.max_train_samples), LABEL_COLUMN)
    test_dataset = filter_labeled(load_split(args.test_split, args.max_test_samples), LABEL_COLUMN)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Speaker verification requires labeled train and test splits with client_id values.")

    full_dataset = concatenate_datasets([train_dataset, test_dataset])
    label2id, id2label = build_label_mapping(full_dataset[LABEL_COLUMN])
    print(f"speaker identities ({len(label2id)}): {len(label2id)} unique client_id values")
    print(f"combined samples: {len(full_dataset)}")

    embeddings, labels = load_or_compute_embeddings(
        full_dataset, LABEL_COLUMN, label2id, args.model, args.layer, args.device, args.batch_size, args.seed, args.cache_dir
    )

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

    print(f"speaker calibration ROC AUC: {calibration_summary['roc_auc']:.4f}")
    print(f"speaker calibration EER: {calibration_summary['eer']:.4f}")
    print(f"speaker threshold: {calibration_summary['threshold']:.4f}")
    print(f"speaker evaluation ROC AUC: {evaluation_summary['roc_auc']:.4f}")
    print(f"speaker evaluation EER: {evaluation_summary['eer']:.4f}")
    print(f"speaker evaluation accept accuracy: {evaluation_accept_accuracy:.4f}")
    print(f"speaker evaluation reject rate: {evaluation_reject_rate:.4f}")


if __name__ == "__main__":
    main()