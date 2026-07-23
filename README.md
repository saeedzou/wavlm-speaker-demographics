# WavLM Common Voice Experiments

This workspace contains five evaluation scripts built on top of a shared WavLM embedding pipeline for Common Voice 17 English.

## What This Does

- Extracts WavLM embeddings from a selectable hidden layer
- Uses random 3-second crops from each utterance when embeddings are computed
- Runs a downstream classifier for gender, accent, and age
- Runs open-set speaker verification using `client_id`
- Uses a validation split for early stopping
- Uses cosine annealing during supervised training

## Files

- `wavlm_common.py` - shared dataset, embedding, training, and verification utilities
- `gender_classification.py` - gender classification
- `accent_classification.py` - accent classification
- `age_classification.py` - age classification
- `speaker_verification.py` - open-set speaker verification
- `speaker_grl.py` - multitask speaker training with gradient reversal for age, gender, and accent

## Requirements

Install the Python packages needed by the scripts:

- `torch`
- `torchaudio`
- `datasets`
- `numpy`
- `scikit-learn`
- `tqdm`
- `transformers`

## Dataset

The scripts expect the Hugging Face dataset:

`saeedzou/common-voice-17-en-age-gender-accent`

The supervised scripts use the `gender`, `accent`, and `age` columns. The speaker script uses `client_id`.

## Running The Scripts

Each script supports CLI flags for the WavLM model and the hidden layer:

- `--model` to choose a WavLM checkpoint
- `--layer` to choose a hidden state index
- `--loss` to choose `ce` or `focal` for the classification scripts
- `--focal_gamma` to tune focal loss when `--loss focal` is selected
- `--trainable_layers` in `speaker_grl.py` to choose how many top WavLM layers are fine-tuned; use `-1` to freeze the encoder
- `--alpha`, `--beta`, and `--gamma` in `speaker_grl.py` to weight the adversarial age, gender, and accent losses
- `--speaker_margin` and `--speaker_scale` in `speaker_grl.py` to tune the AAM-Softmax speaker head
- `--grl_ramp_steps` in `speaker_grl.py` to ramp the gradient reversal strength from `0` to `1`
- `--train_split`, `--val_split`, and `--test_split`
- `--max_train_samples`, `--max_val_samples`, and `--max_test_samples`
- `--device`, `--batch_size`, and classifier hyperparameters
- `--metrics_csv` to write per-epoch metrics and the final test summary to a CSV file

### Gender

```bash
python gender_classification.py --model microsoft/wavlm-base-plus --layer -1 --loss focal
```

### Accent

```bash
python accent_classification.py --model microsoft/wavlm-base-plus --layer -1 --loss focal
```

### Age

```bash
python age_classification.py --model microsoft/wavlm-base-plus --layer -1 --loss focal
```

### Speaker Verification

```bash
python speaker_verification.py --model microsoft/wavlm-base-plus --layer -1
```

### Speaker GRL

```bash
python speaker_grl.py --model microsoft/wavlm-base-plus --layer -1 --trainable_layers 4 --alpha 1.0 --beta 1.0 --gamma 1.0
```

## Typical Workflow

1. Pick the task script you want to run.
2. Select a WavLM checkpoint with `--model`.
3. Select a hidden layer with `--layer`.
4. Optionally limit the data with `--max_train_samples` and `--max_test_samples`.
5. Run the script and wait for the printed epoch-level validation logs.
6. Review the final test result.

## Training Behavior

For the supervised tasks:

- The training split is divided into train and validation subsets.
- The dataset validation split is used directly for early stopping.
- Validation class distributions are printed alongside train and test distributions.
- Early stopping uses validation accuracy.
- The test split is used only for the final report.
- Each supervised run also writes a CSV with per-epoch metrics and a final summary row.
- Learning-rate annealing is enabled through cosine annealing.

For speaker verification:

- Each run writes calibration and evaluation metrics to a CSV file.

For speaker verification:

- `client_id` is treated as the identity label.
- The script builds open-set verification trials.
- Final output reports verification metrics such as ROC AUC and EER.

For speaker GRL:

- `client_id` is treated as the main speaker identity label.
- The WavLM encoder can be partially unfrozen by selecting the number of trainable top layers.
- The speaker branch uses AAM-Softmax, while age, gender, and accent heads are trained through gradient reversal.
- A single run reports speaker accuracy plus the adversarial task accuracies and writes them to CSV.

## Notes

- The scripts use random cropping when embeddings are computed, so rerunning a script can produce a different crop.
- For smaller subsets, stratified validation splitting may fail if classes are too sparse.
- If you want faster smoke tests, start with small `--max_train_samples` and `--max_test_samples` values.
