# Experimental Setup — Assignment 1

This directory documents the full experimental setup, optimization protocol, and environmental configurations for **Assignment 1: Foundations of Deep Learning & Image Classification on Fashion-MNIST** (Course CO3133, Semester 261, Group 07).

---

## 1. Compliance with Handbook Protocol (Section 3.3 & Section 12.2)

To ensure **strict experimental fairness**, all candidate architectures were trained under a unified, identical protocol. Confounding variables (such as divergent optimizers, varying batch sizes, or dissimilar scheduler policies) were deliberately controlled.

### Unified Experimental Protocol Table

| # | Hyperparameter / Setup | Value / Configuration | Design Rationale & Protocol Compliance |
| :-: | :--- | :--- | :--- |
| **1** | **Optimizer** | `torch.optim.Adam` | Uniform optimizer family applied across all architectures to satisfy Handbook Section 12.2 fairness constraints ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$). |
| **2** | **Learning Rate** | Initial `lr = 1e-3` (0.001); minimum `min_lr = 1e-6` | Standard robust baseline for Adam on normalized Fashion-MNIST pixel tensors; dynamically reduced upon plateauing. |
| **3** | **Batch Size** | `128` | Constant across `train_loader`, `val_loader`, and `test_loader` (with `drop_last=False` and `num_workers=2`). Balances gradient estimation stability with GPU throughput. |
| **4** | **Epochs** | Maximum `200` epochs (governed by early stopping) | Upper bound ensuring complete convergence. Actual epochs executed:<br>• **Linear:** 38 epochs<br>• **MLP:** 104 epochs<br>• **CNN:** 86 epochs |
| **5** | **Scheduler** | `ReduceLROnPlateau(mode="min", factor=0.5, patience=3, threshold=1e-4)` | Monitors validation loss; decays learning rate by 50% ($factor=0.5$) when validation loss ceases to improve for 3 consecutive epochs, enabling fine-grained convergence near local minima. |
| **6** | **Regularization** | • **Linear:** None (unregularized baseline).<br>• **MLP:** `Dropout(p=0.3)` after each hidden ReLU layer.<br>• **CNN:** `BatchNorm2d` across all conv blocks + spatial `Dropout2d(0.25, 0.3)` in conv features + `Dropout(p=0.3)` in classification head. | Prevents co-adaptation of hidden features and mitigates overfitting, with Batch Normalization providing additional implicit regularization. |
| **7** | **Early Stopping** | `patience = 8` epochs; tolerance threshold $\Delta = 10^{-4}$ on `val_loss` | Automatically halts training if `val_loss < best_val_loss - 1e-4` is not achieved within 8 consecutive epochs, saving computation and mitigating late-stage over-adaptation. |
| **8** | **Checkpoint Criterion** | Minimum Validation Loss (`best_val_loss`) | The exact model state (`state_dict`) yielding the lowest validation loss is deep-copied in memory and persisted to Google Drive (`CKPT_DIR`). Automatically restored for final test evaluation. |
| **9** | **Seed** | `SEED = 42` (`N_SEEDS = 1` primary run; full seed protocol) | Deterministic random seed set across `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)` to ensure bit-exact reproducibility. |
| **10** | **Hardware** | NVIDIA Tesla T4 GPU (15,360 MiB VRAM), Google Colab instance | OS: Linux 6.6 x86_64, PyTorch 2.11.0+cu128, Python 3.13.15. Device accelerated via CUDA (`device = "cuda"`). |
| **11** | **Mixed Precision** | Not used (Standard Single Precision FP32 across all models) | Guarantees deterministic arithmetic, prevents numerical underflow in lightweight models, and enables unskewed inference latency benchmarking. |
| **12** | **Training Time** | • **Linear:** ~1,142.2 s (~19.0 min)<br>• **MLP:** ~1,845.3 s (~30.8 min)<br>• **CNN:** ~1,811.7 s (~30.2 min)<br>*Total benchmark run: ~4,799.2 s (~1h 20m)* | Wall-clock execution time measured from initialization through early stopping trigger on identical hardware under GPU acceleration. |
| **13** | **Hyperparameter Selection Method** | Matched parameter budget target (~230K–245K params) & manual tuning | Layer dimensions and channel capacities were systematically sized to adhere to a matched capacity target of ~230K–245K parameters (MLP: 242,762; CNN: 231,626) to enable controlled, fair comparison across architectural families under comparable representation capacity. Optimization hyperparameters (learning rate, batch size, scheduler) were held strictly uniform across models to satisfy Handbook Section 12.2 fairness constraints. |

---

## 2. Training Loop & Validation Protocol

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=1e-6
)
```

Each epoch proceeds through:
1. **Forward and Backward Pass (`fit`)**: Mini-batches of 128 images compute loss and gradients via Adam.
2. **Validation Pass (`evaluate`)**: Computes unweighted Top-1 Accuracy, Precision, Recall, Macro-F1, and CrossEntropy Loss over the unseen validation split.
3. **Plateau Scheduler Step**: `scheduler.step(val_loss)` triggers decay if validation loss stagnates for 3 epochs.
4. **Early Stopping & Checkpointing**: If `val_loss < best_val_loss - 1e-4`, the checkpoint is updated; if no progress occurs for 8 epochs, training terminates and the best weights are restored.

---

## 3. Reproducibility Guarantee

All experiments are executed using fixed seed initialization:
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
Checkpoints containing weights, history curves, test metrics, and test prediction arrays are stored in `checkpoints/` and can be reloaded without retraining.
