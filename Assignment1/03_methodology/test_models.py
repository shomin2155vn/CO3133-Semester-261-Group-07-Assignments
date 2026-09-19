"""
Unit test and architectural verification script for all 5 mandatory models.
Runs with dummy input batch corresponding to Fashion-MNIST:
    Input shape: (batch_size=4, channels=1, height=28, width=28)
    Target shape: (batch_size=4,) with class indices in [0, 9]
"""

import sys
import torch
import torch.nn as nn
from models import (
    LinearClassifier,
    MLP,
    CustomCNN,
    ImageRNN,
    ImageTransformer,
    get_model,
    count_parameters
)
from pipeline import get_loss_function, prediction_step


def test_all_architectures():
    print("=" * 80)
    print("ASSIGNMENT 1: 5 MANDATORY ARCHITECTURES VERIFICATION")
    print("Handbook Reference: CO3133 Semester 261 - Section 3.2 & Section 11.1")
    print("=" * 80)

    # Dummy batch matching Fashion-MNIST dimensions
    B, C, H, W = 4, 1, 28, 28
    dummy_input = torch.randn(B, C, H, W)
    dummy_targets = torch.tensor([0, 3, 7, 9], dtype=torch.long)
    criterion = get_loss_function()

    models_dict = {
        "1. Linear / Softmax Classifier": LinearClassifier(num_classes=10),
        "2. Multilayer Perceptron (MLP)": MLP(num_classes=10),
        "3. Convolutional Neural Network (CNN)": CustomCNN(in_channels=1, num_classes=10),
        "4. Sequence Model (Bidirectional LSTM)": ImageRNN(input_size=28, hidden_size=128, num_layers=2, bidirectional=True, rnn_type="lstm"),
        "5. Vision Transformer (ViT)": ImageTransformer(in_channels=1, patch_size=4, embed_dim=64, depth=4, num_heads=4, num_classes=10),
    }

    results = []
    print(f"\nTesting with dummy input batch: {list(dummy_input.shape)}")
    print(f"{'Model Name':<40} | {'Param Count':<12} | {'Logits Shape':<14} | {'Loss':<8} | {'Status'}")
    print("-" * 86)

    for name, model in models_dict.items():
        model.train()
        params = count_parameters(model)["total"]
        
        # Forward pass
        logits = model(dummy_input)
        assert logits.shape == (B, 10), f"Error: expected shape {(B, 10)}, got {logits.shape}"

        # Loss calculation
        loss = criterion(logits, dummy_targets)

        # Backward pass check
        model.zero_grad()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, f"Error: no gradients computed for {name}"

        # Post-processing prediction check
        preds, probs = prediction_step(logits)
        assert preds.shape == (B,)
        assert probs.shape == (B, 10)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)

        results.append({
            "name": name,
            "params": params,
            "loss": loss.item()
        })
        print(f"{name:<40} | {params:<12,d} | {str(list(logits.shape)):<14} | {loss.item():<8.4f} | PASSED [OK]")

    print("-" * 86)
    print("ALL 5 MANDATORY MODELS VERIFIED SUCCESSFULLY!\n")


if __name__ == "__main__":
    test_all_architectures()
