from collections import Counter

from wavlm_common import (
    add_classification_loss_args,
    build_arg_parser,
    build_label_mapping,
    filter_labeled,
    filter_to_labels,
    load_embeddings,
    load_split,
    set_seed,
    train_eval,
)


RAW_AGE_COLUMN = "age"
LABEL_COLUMN = "age_bin"
AGE_BINS = {
    "teens": "young",
    "twenties": "young",
    "thirties": "young",
    "fourties": "middle",
    "fifties": "middle",
    "sixties": "middle",
    "seventies": "old",
    "eighties": "old",
    "nineties": "old",
}


def bin_age_label(example):
    return {LABEL_COLUMN: AGE_BINS.get(example[RAW_AGE_COLUMN])}


def main():
    parser = build_arg_parser("WavLM age classification on Common Voice 17 English")
    add_classification_loss_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    train_dataset = filter_labeled(
        load_split(args.train_split, args.max_train_samples, args.dataset_repo), RAW_AGE_COLUMN
    )
    val_dataset = filter_labeled(
        load_split(args.val_split, args.max_val_samples, args.dataset_repo), RAW_AGE_COLUMN
    )
    test_dataset = filter_labeled(load_split(args.test_split, args.max_test_samples, args.dataset_repo), RAW_AGE_COLUMN)

    if len(train_dataset) == 0:
        raise ValueError("No labeled training examples were found for age classification.")

    train_dataset = filter_labeled(train_dataset.map(bin_age_label), LABEL_COLUMN)
    val_dataset = filter_labeled(val_dataset.map(bin_age_label), LABEL_COLUMN)
    test_dataset = filter_labeled(test_dataset.map(bin_age_label), LABEL_COLUMN)

    label_list = ["young", "middle", "old"]
    val_dataset = filter_to_labels(val_dataset, LABEL_COLUMN, label_list)
    if len(val_dataset) == 0:
        raise ValueError("No labeled validation examples remain after binning age labels.")
    test_dataset = filter_to_labels(test_dataset, LABEL_COLUMN, label_list)
    if len(test_dataset) == 0:
        raise ValueError("No labeled evaluation examples remain after binning age labels.")

    label2id, id2label = build_label_mapping(label_list)
    print(f"age classes ({len(label2id)}): {label_list}")
    print(f"train samples: {len(train_dataset)} | val samples: {len(val_dataset)} | test samples: {len(test_dataset)}")
    train_counts = Counter(train_dataset[LABEL_COLUMN])
    val_counts = Counter(val_dataset[LABEL_COLUMN])
    test_counts = Counter(test_dataset[LABEL_COLUMN])
    print(
        "train class distribution: "
        + ", ".join(f"{label}: {train_counts.get(label,0)}" for label in label_list)
    )
    print(
        "val class distribution: "
        + ", ".join(f"{label}: {val_counts.get(label,0)}" for label in label_list)
    )
    print(
        "test class distribution: "
        + ", ".join(f"{label}: {test_counts.get(label,0)}" for label in label_list)
    )

    X_train, y_train = load_embeddings(
        train_dataset, LABEL_COLUMN, label2id, args.model, args.layer, args.device, args.batch_size
    )
    X_val, y_val = load_embeddings(
        val_dataset, LABEL_COLUMN, label2id, args.model, args.layer, args.device, args.batch_size
    )
    X_test, y_test = load_embeddings(
        test_dataset, LABEL_COLUMN, label2id, args.model, args.layer, args.device, args.batch_size
    )

    acc, report, metrics_csv = train_eval(
        X_train, y_train, X_val, y_val, X_test, y_test, id2label, args, "age_classification"
    )
    print(f"age accuracy: {acc:.4f}")
    print(report)
    print(f"age metrics csv: {metrics_csv}")


if __name__ == "__main__":
    main()