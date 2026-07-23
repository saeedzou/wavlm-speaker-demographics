from wavlm_common import (
    add_classification_loss_args,
    build_arg_parser,
    build_label_mapping,
    filter_labeled,
    filter_to_labels,
    get_label_list,
    load_or_compute_embeddings,
    load_split,
    set_seed,
    train_eval,
)


LABEL_COLUMN = "accent"


def main():
    parser = build_arg_parser("WavLM accent classification on Common Voice 17 English")
    add_classification_loss_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    train_dataset = filter_labeled(
        load_split(args.train_split, args.max_train_samples, args.dataset_repo), LABEL_COLUMN
    )
    test_dataset = filter_labeled(load_split(args.test_split, args.max_test_samples, args.dataset_repo), LABEL_COLUMN)

    if len(train_dataset) == 0:
        raise ValueError("No labeled training examples were found for accent classification.")

    label_list = get_label_list(train_dataset, LABEL_COLUMN)
    test_dataset = filter_to_labels(test_dataset, LABEL_COLUMN, label_list)
    if len(test_dataset) == 0:
        raise ValueError("No labeled evaluation examples remain after filtering to training labels.")

    label2id, id2label = build_label_mapping(label_list)
    print(f"accent classes ({len(label2id)}): {label_list}")
    print(f"train samples: {len(train_dataset)} | test samples: {len(test_dataset)}")

    X_train, y_train = load_or_compute_embeddings(
        train_dataset, LABEL_COLUMN, label2id, args.model, args.layer, args.device, args.batch_size, args.seed, args.cache_dir
    )
    X_test, y_test = load_or_compute_embeddings(
        test_dataset, LABEL_COLUMN, label2id, args.model, args.layer, args.device, args.batch_size, args.seed, args.cache_dir
    )

    acc, report, metrics_csv = train_eval(X_train, y_train, X_test, y_test, id2label, args, "accent_classification")
    print(f"accent accuracy: {acc:.4f}")
    print(report)
    print(f"accent metrics csv: {metrics_csv}")


if __name__ == "__main__":
    main()