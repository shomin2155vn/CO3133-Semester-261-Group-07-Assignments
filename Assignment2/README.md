# Reproducing the Assignment 2 Food-11 results

Use [Assignment2.ipynb](Assignment2.ipynb) to rebuild the cleaned Food-11 splits, evaluate the saved SimpleCNN and ResNet-18 checkpoints, and regenerate the result tables and figures. The same notebook can train all three models on Google Colab. The [checkpoint archive](checkpoints/) and the [selected error-analysis figures](04_error_analysis/) are in this repository.

| Goal | Start here |
| --- | --- |
| Rebuild the notebook tables and figures using the published weights | [Run the notebook with saved checkpoints](#1-run-the-notebook-with-saved-checkpoints) |
| Train the models again from their initial weights | [Retrain on Colab](#4-retrain-on-colab) |


## 1. Run the notebook with saved checkpoints

1. Open [Assignment2.ipynb](Assignment2.ipynb) in Google Colab. Its setup uses `/content` and the training mode uses the Colab Drive integration.
2. Use a GPU runtime for the full report pass, especially for inference timing. The evaluation code can fall back to CPU. A CPU run may take longer and produces timings for that machine.
3. Run the installation cell. It installs KaggleHub, NumPy, pandas, Pillow, Matplotlib, tqdm, scikit-learn, SciPy, IPython, and ImageHash. The runtime also needs compatible PyTorch and torchvision installations. The notebook uses Git to fetch artifacts, KaggleHub to download Food-11, and torchvision to load the ImageNet ResNet-18 initialization.
4. Set these values in the import and configuration cell **before** running it.

```python
ARTIFACT_SOURCE = "github"
TRAIN_MODELS = False
PIPELINE_VERSION = "food11-256x256-Augmented-fp32"
SINGLE_SEED = 42
RECOMPUTE_TEST = True
SAVE_REPORTS = True
```

The notebook's default is `SAVE_REPORTS = False`. Setting it to `True` writes the report CSVs and PNGs. The data audit, split manifests, and environment file are written during setup in either case.

The GitHub mode checks all nine model/seed runs for the selected variant and copies the four required test CSVs per run into a writable output directory.

| Variant | `PIPELINE_VERSION` | Training flip probability |
| --- | --- | ---: |
| Main experiment | `food11-256x256-Augmented-fp32` | 0.5 |
| Ablation | `food11-256x256-No_Augmented-fp32` | 0 |

Use a separate runtime for each variant, or change `PIPELINE_VERSION` and rerun from the configuration cell.

## 2. Rebuild the data split

Run **Dataset Installation**, **EDA**, **Leakage Prevention**, and **Data Preprocessing** in order. The notebook downloads `trolukovich/food11-image-dataset` version `1` and expects the official `training/`, `validation/`, and `evaluation/` folders. Keep the original folders and labels.

The image scan records byte-level MD5 and perceptual hashes. The candidate audit uses a pHash Hamming distance of at most 10. The removal pass applies MD5 and then pHash at distance at most 5, checking within each split before checking across splits. Retention priority is train, validation, then test. The notebook also drops `training/Soup/195.jpg` and `training/Egg/425.jpg` after manual review.

| Split | Original images | Retained images |
| --- | ---: | ---: |
| Train | 9,866 | 9,846 |
| Validation | 3,430 | 3,364 |
| Test | 3,347 | 2,770 |
| Total | 16,643 | 15,980 |

 All models use RGB inputs resized with the aspect ratio preserved, padded in black to 256 × 256, and normalized with the ImageNet mean and standard deviation. The optional horizontal flip applies only to training.

## 3. Regenerate the report outputs

To reproduce the report from saved checkpoints, run the notebook from top to bottom. 

`RUN_THREE_SEEDS` determines which seeds are validated and recorded in `environment.json`. `THREE_SEEDS` lists the configured seeds and is also used to locate GitHub artifacts and assign colors to learning curves. Training in cells 33–35 is controlled by `SINGLE_SEED`.

For parallel retraining, open **three separate Colab sessions** with the same `PIPELINE_VERSION` and set `ARTIFACT_SOURCE = "drive"` and `TRAIN_MODELS = True` in each. Assign `SINGLE_SEED = 42`, `43`, and `44` to the three sessions, respectively. Each session trains all three models for its assigned seed and saves the runs under distinct model-and-seed names. Running the sessions concurrently reduces the elapsed time to finish the experiment.

After all nine runs for a variant are complete, run **Three Seed** in one session to aggregate the saved results. This section uses `REPORT_SEEDS = [42, 43, 44]` and can load all nine runs with `RUN_THREE_SEEDS = False`.

Keep `RECOMPUTE_TEST = True` for a fresh notebook run. The published prediction CSVs can contain paths such as `/kaggle/input/food11-image-dataset/evaluation/...`, whereas Colab uses a different local path. The quantitative report's saved-prediction mode checks absolute paths against the current test set. Use the CSV reconstruction below to inspect the archived results without rerunning inference.

In GitHub mode the output directory is `/content/food11_reproduce/<PIPELINE_VERSION>/`. With `SAVE_REPORTS = True`, look for:

| Path | Contents |
| --- | --- |
| `environment.json`, `splits/` | Runtime details, cleaned manifests, and split hash |
| `tables/single_seed42_*.csv`, `tables/three_seeds_42_43_44_*.csv` | Run-level and aggregate metrics, training summaries, per-class results, and confusion matrices |
| `tables/<run>_scored_predictions.csv`, `tables/<run>_qualitative_examples.csv` | Test-image scores and selected qualitative examples |
| `figures/*_learning_curves.png`, `figures/*_confusion_matrices.png` | Training and validation plots and test confusion matrices |
| `figures/<run>_correct.png`, `figures/<run>_difficult_correct_candidates.png`, `figures/<run>_failures.png` | Seed-42 example grids |


## 4. Retrain on Colab

1. Use a GPU runtime and set `ARTIFACT_SOURCE = "drive"` and `TRAIN_MODELS = True` in the configuration cell. Choose the desired `PIPELINE_VERSION` and run the setup through the `run_experiment` definition.
2. The Drive mode writes to `/content/drive/MyDrive/Deep_Learning/Assignment2/<PIPELINE_VERSION>/`. Use a fresh subfolder under Drive if you need to retain previous runs. A training run writes its checkpoint, history, config, and evaluation CSVs under the same model/seed name on every rerun.


The notebook uses FP32, 25 epochs, batch size 128, AdamW with weight decay 0.01, and unweighted cross-entropy. The learning rate starts at 0.001 for SimpleCNN and the classification heads, with 0.0001 for the backbone during full fine-tuning. The scheduler follows validation loss. The selected checkpoint maximizes validation Macro-F1, with lower validation loss breaking ties. Test evaluation uses that checkpoint.

The archived `logs/*_config.json` files record the effective settings and software versions. 


