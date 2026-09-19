"""
CO3133 - Deep Learning and Its Applications | Semester 261 | Assignment 1
Group 07: Hoàng Xuân Bách, Nguyễn Việt Hùng, Lê Nguyễn Gia Phúc
Main responsibility: Lê Nguyễn Gia Phúc (Model architecture, training)

Architectures for Fashion-MNIST classification (1x28x28 grayscale, 10 classes)
adhering to Course Project Handbook (Section 3.2 & Section 11):
1. Linear / Softmax Classifier
2. Multilayer Perceptron (MLP)
3. Convolutional Neural Network (CNN - custom designed)
4. Sequence Model (Image LSTM / GRU)
5. Vision Transformer (ViT / Patch-based Transformer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Linear / Softmax Classifier
# Handbook rule:
# - Flatten each image into a vector.
# - Use a linear layer to produce logits.
# - Use cross-entropy loss.
# - Do NOT apply softmax before CrossEntropyLoss.
# ==============================================================================
class LinearClassifier(nn.Module):
    """
    Linear / Softmax Classifier for Fashion-MNIST.
    Maps flattened input x in R^(1*28*28) = R^784 directly to 10 class logits:
        z = W * x + b
    Note: Returns raw unnormalized logits. Softmax is omitted here because
    PyTorch nn.CrossEntropyLoss integrates LogSoftmax and NLLLoss for numerical stability.
    """
    def __init__(self, in_features: int = 784, num_classes: int = 10):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) or (B, 784)
        x_flat = self.flatten(x)  # (B, 784)
        logits = self.fc(x_flat)  # (B, 10)
        return logits


# ==============================================================================
# 2. Multilayer Perceptron (MLP)
# Handbook rule:
# - At least one hidden layer.
# - Explain activation functions and any regularization used.
# ==============================================================================
class MLP(nn.Module):
    """
    Multilayer Perceptron (Feedforward Neural Network) with 2 hidden layers.
    Architecture:
        Input: 784 -> Linear(256) -> BatchNorm1d -> ReLU -> Dropout(0.2)
                   -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.2)
                   -> Linear(10) -> Logits
    Regularization:
        - BatchNorm1d: Stabilizes internal covariate shift and accelerates convergence.
        - Dropout (p=0.2): Prevents co-adaptation of hidden units and reduces overfitting.
    """
    def __init__(
        self,
        in_features: int = 784,
        hidden_dims: tuple = (256, 128),
        num_classes: int = 10,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            ])
            prev_dim = h_dim
        
        # Classification head outputting raw logits
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = self.flatten(x)
        return self.network(x_flat)


# ==============================================================================
# 3. Convolutional Neural Network (CNN)
# Handbook rule:
# - Design the architecture yourselves (not calling a pretrained model).
# - Explain convolution, pooling, and feature maps.
# ==============================================================================
class CustomCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network designed for Fashion-MNIST.
    Leverages spatial inductive bias (local receptive fields and translation equivariance).
    
    Architecture:
        Stage 1 (Low-level features: edges, textures):
            Conv2d(1 -> 32, kernel=3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(32 -> 32, kernel=3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2, 2) -> Spatial dim: 28x28 -> 14x14
            Dropout2d(0.1)

        Stage 2 (Mid-level features: contours, local garment parts):
            Conv2d(32 -> 64, kernel=3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(64 -> 64, kernel=3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2, 2) -> Spatial dim: 14x14 -> 7x7
            Dropout2d(0.15)

        Stage 3 (High-level semantic features: global garment silhouette):
            Conv2d(64 -> 128, kernel=3, pad=1) -> BatchNorm2d -> ReLU
            AdaptiveAvgPool2d((2, 2)) -> Feature map: (B, 128, 2, 2)
            Dropout(0.2)

        Classifier Head:
            Flatten -> Linear(128*2*2 = 512 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 10)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.10)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Dropout(p=0.20)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        logits = self.classifier(out)
        return logits


# ==============================================================================
# 4. Sequence Model (Image LSTM / GRU)
# Handbook rule:
# - Represent each image as a sequence of rows, columns, or patches.
# - Explain timestep definition, input size, and hidden representation.
# ==============================================================================
class ImageRNN(nn.Module):
    """
    Recurrent Neural Network (LSTM / GRU) for 2D Image Classification.
    
    Sequence Formulation:
        Each 28x28 grayscale image is modeled as a temporal sequence of rows:
        - Timesteps (T): 28 timesteps (row by row from top to bottom).
        - Input dimension per timestep (d_in): 28 pixels per row.
        - Sequence tensor shape: (Batch_Size, T=28, d_in=28).
        
    Hidden Representation:
        A multi-layer Bidirectional LSTM (or GRU) aggregates context across rows.
        At each step t, the cell updates its memory state based on current row and previous row context.
        The final representation concatenates forward and backward hidden states,
        followed by an MLP head to output logits.
    """
    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0.0
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Use 'lstm' or 'gru'.")

        feat_dim = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) -> squeeze channel to get (B, 28, 28)
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, T=28, D=28)
        
        # rnn_out: (B, T=28, num_directions * hidden_size)
        if self.rnn_type == "lstm":
            rnn_out, (h_n, c_n) = self.rnn(x)
        else:
            rnn_out, h_n = self.rnn(x)

        # Extract features from final states or mean-pooling across all rows
        # Combining forward state from last step and backward state from first step:
        if self.bidirectional:
            # h_n shape: (num_layers * 2, B, hidden_size)
            forward_last = h_n[-2, :, :]
            backward_last = h_n[-1, :, :]
            feat = torch.cat([forward_last, backward_last], dim=1)  # (B, 2 * hidden_size)
        else:
            feat = h_n[-1, :, :]  # (B, hidden_size)

        logits = self.classifier(feat)
        return logits


# ==============================================================================
# 5. Vision Transformer (ViT / Patch-based Transformer)
# Handbook rule:
# - Represent each image as rows/columns or patches.
# - Include token embedding/projection and positional encoding.
# - Explain attention inputs and outputs.
# ==============================================================================
class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects each patch to embedding dim D.
    For Fashion-MNIST: image (1, 28, 28), patch_size = 4:
        - Number of patches: (28/4) * (28/4) = 7 * 7 = 49 patches.
        - Patch flat dimension: 1 * 4 * 4 = 16 pixels.
        - Linear projection: R^16 -> R^d_model.
    Implemented efficiently using Conv2d with kernel_size = stride = patch_size.
    """
    def __init__(self, in_channels: int = 1, patch_size: int = 4, embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) * (28 // patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        # proj(x): (B, embed_dim, 7, 7)
        # flatten spatial: (B, embed_dim, 49) -> transpose to (B, 49, embed_dim)
        x_proj = self.proj(x).flatten(2).transpose(1, 2)
        return x_proj


class ImageTransformer(nn.Module):
    """
    Compact Vision Transformer (ViT) for Fashion-MNIST.
    
    Components:
        1. Patch Projection: Transforms 49 patches of 4x4 pixels into d_model embeddings.
        2. [CLS] Token: Prepend learnable classification token to sequence (length = 49 + 1 = 50).
        3. Positional Encoding: Learnable 1D tensor added to retain 2D spatial arrangement.
        4. Transformer Encoder: L layers of Multi-Head Self-Attention (MHSA) and MLP blocks
           with LayerNorm and residual skip connections.
           MHSA Attention Mechanism:
               Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        5. MLP Classification Head: LayerNorm on [CLS] token followed by Linear layer.
    """
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 4,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_classes: int = 10,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable [CLS] token and 1D positional encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)
        
        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False
        )
        
        # Norm and Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        # Truncated normal for embeddings, Xavier for linear
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # (B, num_patches, embed_dim)
        tokens = self.patch_embed(x)
        
        # Expand and concatenate [CLS] token: (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional encodings
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        
        # Pass through Transformer Encoder
        encoded = self.encoder(tokens)  # (B, num_patches + 1, embed_dim)
        
        # Extract representation of [CLS] token (index 0)
        cls_rep = encoded[:, 0]
        cls_rep = self.norm(cls_rep)
        
        logits = self.head(cls_rep)  # (B, num_classes)
        return logits


# ==============================================================================
# Helper Factory & Parameter Counter
# ==============================================================================
def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate any of the 5 mandatory models.
    """
    name = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if name in ["linear", "linearclassifier", "softmax", "softmaxclassifier"]:
        return LinearClassifier(**kwargs)
    elif name in ["mlp", "multilayerperceptron"]:
        return MLP(**kwargs)
    elif name in ["cnn", "customcnn", "convolutionalneuralnetwork"]:
        return CustomCNN(**kwargs)
    elif name in ["lstm", "rnn", "gru", "imagernn", "imagelstm"]:
        return ImageRNN(**kwargs)
    elif name in ["transformer", "vit", "visiontransformer", "imagetransformer"]:
        return ImageTransformer(**kwargs)
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Options: 'linear', 'mlp', 'cnn', 'lstm'/'gru', 'transformer'."
        )


# ==============================================================================
# Model Aliases matching notebook naming conventions
# ==============================================================================
SimpleCNN = CustomCNN
RowSequenceGRU = ImageRNN
RowSequenceLSTM = ImageRNN
PatchTransformer = ImageTransformer


def count_parameters(model: nn.Module) -> dict:
    """
    Returns parameter statistics for a PyTorch module.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }
