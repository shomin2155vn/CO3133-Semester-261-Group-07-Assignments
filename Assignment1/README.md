# Assignment 1 — Fashion-MNIST Reproducibility

All experiments are implemented in **`Assignment1/Assignment1.ipynb`**.

## 1. Installation, dependency versions, and hardware

Open the notebook in Google Colab and select **Runtime → Change runtime type → GPU**. The reference environment is:

| Component | Value |
| --- | --- |
| Hardware | NVIDIA Tesla T4 GPU |
| Platform / precision | Google Colab, Linux x86_64 / FP32 |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| Torchvision | 0.26.0+cu130 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Pillow | 11.3.0 |

Download `fmnist_install_config.json` from an experiment directory under `Reproducibility/multiseed_42_43_44/results/reproducibility/` and upload it to Colab as:

```text
/content/fmnist_install_config.json
```

Run the notebook's **first installation cell**. It calls `python -m pip install` using the recorded versions and CUDA wheel index. If imported packages change, restart the runtime before continuing. Without the JSON file, the cell installs unpinned dependencies.

## 3. Dataset preparation

Run the data-loading, EDA, and preprocessing cells in order. Fashion-MNIST is downloaded automatically to `./data` using Torchvision.

| Partition | Images | Images per class |
| --- | ---: | ---: |
| Training | 48,000 | 4,800 |
| Validation | 12,000 | 1,200 |
| Official test | 10,000 | 1,000 |

The official 60,000-image training pool is split with stratified `train_test_split(test_size=0.2, random_state=42)`. Normalization uses only the training split: mean `0.2862528489520806`, standard deviation `0.3531738779381928`.

Training applies random rotation (−10° to +10°), horizontal flip (`p=0.5`), random crop (28×28, padding 2), tensor conversion, and normalization. Validation and test images use only tensor conversion and normalization. The notebook also downloads MNIST for debugging; Fashion-MNIST supplies the reported experiments.

## 4. Configuration files and seed settings

| Configuration | Purpose |
| --- | --- |
| Notebook configuration cell and model definitions | Active training/evaluation settings and architectures |
| `fmnist_install_config.json` | Dependency versions read by the installation cell |
| `experiment_config.json` | Exported protocol, seeds, split signature, normalization, training settings, and environment |
| `<run_id>/model_configuration.json` | Exported architecture, parameter count, and dropout settings |
| `dataset_split.npz` | Ordered training/validation indices, test indices, and split seed |

Use these shared settings:

```python
SPLIT_SEED = 42
TRAIN_SEEDS = [42, 43, 44]
RUN_VARIANTS = ["Baseline", "NoDropout"]
EPOCHS = 200
LR = 1e-3
PATIENCE = 8
NUM_WORKERS = 2
EVAL_BATCH_SIZE = 256
REQUIRE_ALL_SEEDS = True
```

Training batch size is 128. All models use cross-entropy, Adam with no weight decay, and ReduceLROnPlateau on validation loss (factor 0.5, patience 3, absolute threshold `1e-4`, minimum learning rate `1e-6`). Checkpoints are selected by minimum validation loss; early stopping uses patience 8.


## 5. Evaluate existing checkpoints

In the configuration cell, set:

```python
TRAIN_MODELS = False
FORCE_RETRAIN = False
RE_EVALUATE = True
```

In the path-setup cell, set:

```python
USE_GITHUB = True
```

Then run the remaining cells from top to bottom. The notebook clones the repository into `/content/fmnist_repo` and reads checkpoints from:

```text
/content/fmnist_repo/Assignment1/Reproducibility/multiseed_42_43_44/checkpoints
```

The training-group cells skip training. The aggregation cell recomputes test predictions, metrics, and inference time from checkpoint weights, then writes the summaries. Continue through the metadata export and plotting cells to regenerate learning curves, confusion matrices, prediction examples, and paired ablation tables.


`RE_EVALUATE=False` instead uses metrics and predictions stored in checkpoints. With either setting, historical training time and learning curves come from checkpoint history. Fresh inference timings vary between executions.

## 6. Train / reconstruct checkpoints

Set:

```python
TRAIN_MODELS = True
FORCE_RETRAIN = False
RE_EVALUATE = True
```

In the path-setup cell, use `USE_GITHUB=False` to save to your own Google Drive. To separate reconstructed results, change its `MULTISEED_DIR` assignment to:


Run the notebook in order. Its training commands are:

```python
train_group(["Linear", "MLP"])
train_group(["CNN"])
train_group(["LSTM"])
train_group(["Transformer"])
```

Both variants require **27 runs**: 15 Baseline and 12 NoDropout. To run only the main comparison, set `RUN_VARIANTS=["Baseline"]` for 15 runs. NoDropout disables all implemented dropout locations while retaining augmentation, normalization, architecture dimensions, and parameter counts. Linear is excluded because it has no dropout.


## 7. Checkpoints, outputs, and result traceability

Checkpoint filenames follow `{Model}_200K_v4_{Variant}_seed{seed}.pt`. Each is a PyTorch dictionary containing weights, configuration, training history, selected checkpoint information, optimizer/scheduler states, and stored test results. The notebook validates checkpoint configuration before loading.

After aggregation, the export cell creates a new directory:

```text
RESULTS_DIR/reproducibility/exp_<UUID>/
```

| Export | Traceability information |
| --- | --- |
| `experiment_config.json` | Experiment ID, protocol, seeds, split signature, normalization, and environment |
| `dataset_split.npz` | Exact partition indices |
| `experiment_manifest.csv` | Model/variant/seed, run ID, checkpoint path, `checkpoint_sha256`, split signature, and history/config paths |
| `runs_per_seed.csv`, `results_mean_std.csv` | Per-run metrics and mean ± sample standard deviation (`ddof=1`) |
| `<run_id>/model_configuration.json` | Architecture and training configuration |
| `<run_id>/history.csv` | Epoch-level training/validation history |
| `<run_id>/run_metadata.json` | Training environment and evaluation/timing sources |
| `<run_id>/test_predictions.npz` | Labels and predictions used for evaluation |
| `fmnist_install_config.json` | Dependency versions |



Aggregation writes overall/per-class CSVs under `RESULTS_DIR`; the final cell writes paired NoDropout-minus-Baseline differences. Figures go to `FIG_DIR` and `/content/fmnist_multiseed_42_43_44`. Runtime outputs must be copied or downloaded to persist; GitHub mode does not push changes automatically.
