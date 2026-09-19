# 03. Methodology

> **Course:** CO3133 - Deep Learning and Its Applications | Semester 261 | HCMUT  
> **Handbook Alignment:** Section 3.2 (Part 2 — Methodology) & Section 11 (Assignment 1 Technical Requirements)  
> **Person in Charge:** **Lê Nguyễn Gia Phúc** (*Model architecture, training*)

---

## 1. Pipeline Overview

Tuân thủ nghiêm ngặt định dạng pipeline được quy định tại Mục 3.2 của Sổ tay môn học:

$$\text{Raw Data} \longrightarrow \text{Preprocessing} \longrightarrow \text{DataLoader} \longrightarrow \text{Model} \longrightarrow \text{Loss} \longrightarrow \text{Optimization} \longrightarrow \text{Prediction} \longrightarrow \text{Post-processing} \longrightarrow \text{Evaluation}$$

### Chi tiết các giai đoạn:
1. **Raw Data:** Tập dữ liệu gốc **Fashion-MNIST** gồm 70,000 ảnh mức xám ($28 \times 28$, 1 channel, 10 lớp thời trang).
2. **Preprocessing:** Chuyển đổi thành Tensor (`ToTensor()`), chuẩn hóa z-score với mean = 0.2860, std = 0.3530; kết hợp data augmentation (Random Horizontal Flip, Random Crop with padding).
3. **DataLoader:** Phân chia train/val/test cố định seed; batch size = 64, `shuffle=True` trên tập train, `pin_memory=True`.
4. **Model Architecture:** Đưa tensor qua một trong 5 họ mô hình khảo sát (Linear, MLP, CNN, RNN/LSTM, Transformer).
5. **Loss Computation:** Hàm mất mát Cross-Entropy Loss trực tiếp trên unnormalized logits:
   $$\mathcal{L}(\mathbf{z}, y) = -z_y + \log\left(\sum_{j=1}^{C} e^{z_j}\right)$$
6. **Optimization:** Tối ưu hóa trọng số bằng thuật toán AdamW / SGD với Momentum, kết hợp Cosine Annealing Learning Rate Scheduler.
7. **Prediction & Post-processing:** Áp dụng Softmax để lấy phân phối xác suất và $\operatorname{argmax}$ để dự đoán nhãn lớp:
   $$\hat{y} = \arg\max_{c \in \{0, \dots, 9\}} \sigma(\mathbf{z})_c$$
8. **Evaluation:** Đánh giá toàn diện qua Accuracy, Macro-F1, Confusion Matrix, số lượng tham số và thời gian suy luận.

---

## 2. Năm Họ Mô Hình Bắt Buộc (Five Mandatory Architectures)

Toàn bộ 5 mô hình đã được cài đặt độc lập trong [`models.py`](models.py) và kiểm chứng qua [`test_models.py`](test_models.py):

| STT | Họ Mô Hình | Tên Class | Input Representation | Số Tham Số (Params) | Inductive Bias Chính |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | **Linear Classifier** | `LinearClassifier` | Vector phẳng $\mathbf{x} \in \mathbb{R}^{784}$ | **7,850** | Tuyến tính toàn cục; không nắm bắt quan hệ không gian. |
| **2** | **Multilayer Perceptron** | `MLP` | Vector phẳng $\mathbf{x} \in \mathbb{R}^{784}$ | **235,914** | Phi tuyến tính dạng tầng; kết nối dày đặc (fully connected). |
| **3** | **Convolutional Network** | `CustomCNN` | Tensor ảnh $\mathbf{X} \in \mathbb{R}^{1 \times 28 \times 28}$ | **206,378** | Bất biến tịnh tiến (translation equivariance), tính cục bộ (locality). |
| **4** | **Sequence Model** | `ImageRNN` (BiLSTM) | Chuỗi 28 hàng $\times$ 28 chiều ($T=28, D=28$) | **574,282** | Phụ thuộc thứ tự tuần tự không gian theo chiều dọc (top-to-bottom). |
| **5** | **Vision Transformer** | `ImageTransformer` | Chuỗi 49 patches $4\times 4$ + `[CLS]` token | **139,018** | Chú ý tương tác toàn cục (global self-attention) giữa mọi vị trí patch. |

---

## 3. Đặc Tả Chi Tiết Từng Kiến Trúc

### 3.1. Linear / Softmax Classifier
* **Quy tắc Handbook:** Flatten ảnh thành vector 1D; dùng Linear layer sinh logits; dùng Cross-Entropy Loss; **tuyệt đối không áp dụng softmax trước CrossEntropyLoss**.
* **Công thức:** $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, với $\mathbf{W} \in \mathbb{R}^{10 \times 784}$ và $\mathbf{b} \in \mathbb{R}^{10}$.
* **Vai trò:** Đóng vai trò là đường cơ sở tối giản (minimal linear baseline) để đo lường mức độ tách biệt tuyến tính của không gian pixel gốc.

### 3.2. Multilayer Perceptron (MLP)
* **Quy tắc Handbook:** Tối thiểu 1 hidden layer; giải thích hàm kích hoạt và kỹ thuật regularization.
* **Cấu trúc:** 
  $$\mathbf{x} \in \mathbb{R}^{784} \to \text{Linear}(256) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(128) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(10)$$
* **Regularization:**
  - *Batch Normalization:* Giảm internal covariate shift, giúp quá trình lan truyền ngược ổn định.
  - *Dropout ($p=0.2$):* Triệt tiêu hiện tượng đồng thích nghi (co-adaptation) giữa các neuron ẩn.

### 3.3. Convolutional Neural Network (CNN)
* **Quy tắc Handbook:** Tự thiết kế kiến trúc (không chỉ gọi pretrained model từ thư viện); giải thích phép tích chập, pooling và feature maps.
* **Cấu trúc:** Thiết kế 3-stage phân cấp đặc trưng từ cục bộ đến ngữ nghĩa:
  - **Stage 1 (Low-level):** $2 \times [\text{Conv2d}(3\times 3, 32) + \text{BN} + \text{ReLU}] \to \text{MaxPool2d}(2\times 2) \to (32, 14, 14)$.
  - **Stage 2 (Mid-level):** $2 \times [\text{Conv2d}(3\times 3, 64) + \text{BN} + \text{ReLU}] \to \text{MaxPool2d}(2\times 2) \to (64, 7, 7)$.
  - **Stage 3 (High-level):** $[\text{Conv2d}(3\times 3, 128) + \text{BN} + \text{ReLU}] \to \text{AdaptiveAvgPool2d}(2\times 2) \to (128, 2, 2)$.
  - **Classifier Head:** $\text{Flatten} \to \text{Linear}(512, 128) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(128, 10)$.

### 3.4. Sequence Model (Image LSTM / GRU)
* **Quy tắc Handbook:** Biểu diễn ảnh thành chuỗi các hàng/cột/patch; định nghĩa rõ timestep, kích thước đầu vào và hidden representation.
* **Quy ước biểu diễn:** 
  - Ảnh $28 \times 28$ được xem là một chuỗi gồm **$T = 28$ bước thời gian** (tương ứng 28 hàng pixel từ trên xuống dưới).
  - Tại mỗi timestep $t$, đầu vào là vector $\mathbf{x}_t \in \mathbb{R}^{28}$ (toàn bộ giá trị pixel của hàng thứ $t$).
* **Kiến trúc mạng:**
  - Sử dụng **2-layer Bidirectional LSTM** với hidden size $H = 128$.
  - Trạng thái ẩn cuối cùng là sự ghép nối giữa hướng xuôi và hướng ngược: $\mathbf{h} = [\mathbf{h}_{\text{forward}}^{(T)}; \mathbf{h}_{\text{backward}}^{(1)}] \in \mathbb{R}^{256}$.
  - Classifier Head: $\text{Linear}(256, 64) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(64, 10)$.

### 3.5. Vision Transformer (ViT / Patch-based Attention)
* **Quy tắc Handbook:** Biểu diễn ảnh dưới dạng patch; có token embedding/projection và positional encoding; giải thích cơ chế Attention (Inputs: Q, K, V và Outputs).
* **Quy ước Patch:**
  - Kích thước patch $P = 4 \times 4$. Số lượng patch $N = (28/4) \times (28/4) = 49$ patches.
  - Linear Projection qua `Conv2d(1, 64, kernel=4, stride=4)` biến mỗi patch thành token $\mathbf{e}_i \in \mathbb{R}^{64}$.
* **[CLS] Token & Positional Encoding:**
  - Bổ sung learnable `[CLS]` token vào đầu chuỗi: độ dài chuỗi thành $49 + 1 = 50$.
  - Cộng vector vị trí học được $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{50 \times 64}$ để giữ lại thông tin tọa độ không gian 2D.
* **Cơ chế Attention:**
  - Sử dụng 4 tầng `TransformerEncoderLayer` với 4 Attention Heads ($d_k = 16$).
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
  - Logits được dự đoán từ biểu diễn của `[CLS]` token sau khi chuẩn hóa qua `LayerNorm`.

---

## 4. Minh Bạch Thư Viện & Mã Nguồn Tự Xây Dựng (Library Disclosure)

Tuân thủ Mục 3.2 của Sổ tay môn học:
- **Tự thiết kế & cài đặt (Self-implemented):**
  - Toàn bộ 5 lớp mô hình: `LinearClassifier`, `MLP`, `CustomCNN`, `ImageRNN`, `ImageTransformer`.
  - Cơ chế patch projection, khởi tạo `[CLS]` token và learnable positional encoding trong Vision Transformer.
  - Toàn bộ pipeline kết nối, quy trình kiểm thử và đo đếm tham số (`test_models.py`, `pipeline.py`).
- **Thư viện bên ngoài sử dụng:**
  - `torch.nn`: Cung cấp các khối cơ sở (`nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.LSTM`, `nn.TransformerEncoderLayer`, `nn.CrossEntropyLoss`).
  - `torch.optim`: Cung cấp bộ tối ưu `AdamW`, `SGD` và bộ điều chỉnh learning rate `CosineAnnealingLR`.

---

## 5. Hướng Dẫn Chạy Kiểm Thử

Để kiểm tra tính đúng đắn và đếm số lượng tham số của cả 5 mô hình:

```bash
cd Assignment1/03_methodology
python test_models.py
```
