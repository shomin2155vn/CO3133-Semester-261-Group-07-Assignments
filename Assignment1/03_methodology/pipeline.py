"""
CO3133 - Deep Learning and Its Applications | Semester 261 | Assignment 1
Group 07: Hoàng Xuân Bách, Nguyễn Việt Hùng, Lê Nguyễn Gia Phúc

Pipeline Components adhering to Course Project Handbook (Section 3.2):
    Raw data -> preprocessing -> data loader -> model -> loss
    -> optimization -> prediction -> post-processing -> evaluation
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def get_loss_function(label_smoothing: float = 0.0) -> nn.CrossEntropyLoss:
    """
    Returns standard Cross-Entropy Loss.
    
    Handbook Requirement (Section 11.1):
        - "Use cross-entropy loss."
        - "Do not apply softmax before CrossEntropyLoss."
        
    Mathematical formulation:
        For input logits z in R^C and target class y in {0, ..., C-1}:
            L(z, y) = -log( exp(z_y) / sum_{j=1}^C exp(z_j) )
                    = -z_y + log( sum_{j=1}^C exp(z_j) )
    PyTorch nn.CrossEntropyLoss combines LogSoftmax and NLLLoss in a numerically
    stable single operation using the Log-Sum-Exp trick.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9
) -> torch.optim.Optimizer:
    """
    Returns the configured optimizer for model parameter optimization.
    """
    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use 'adamw', 'adam', or 'sgd'.")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 20,
    step_size: int = 7,
    gamma: float = 0.1
):
    """
    Returns learning rate scheduler.
    """
    name = scheduler_name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")


def prediction_step(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Post-processing step from logits to predictions and probabilities:
        probabilities = Softmax(logits, dim=-1)
        predicted_classes = argmax(probabilities, dim=-1)
    """
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    return preds, probs


def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    """
    Computes batch-level accuracy for monitoring during training.
    """
    preds, _ = prediction_step(logits)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    acc = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc
    }
