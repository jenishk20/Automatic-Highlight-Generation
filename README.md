
# ⚽ Key Moments Identification in Soccer Matches

Welcome to our project on **Key Moments Identification from Soccer Videos**. This deep learning-based system is designed to classify short video clips into important soccer events such as goals, fouls, corners, and more. We explored multiple model architectures and preprocessing pipelines to develop a robust video classification system.

---

## 📌 Problem Statement

Manual tagging of soccer highlights is time-consuming and subjective. We aim to automate this process using machine learning by analyzing the visual and temporal patterns in short video clips to classify them into key event types.

---

## 🧠 Approach Overview
We explored three primary deep learning architectures for this task:

1. **ResNet2D + GRU**
2. **ResNet2D + LSTM**
3. **ResNet3D (R3D)**

Each model captures both the **spatial** (what's in the frame) and **temporal** (how things change over time) aspects of the video, but in different ways.

---

## 🧩 Model Architectures

### 1. ResNet2D + GRU
- **Why ResNet2D?**
  - It's a pretrained image classifier that extracts rich spatial features from individual video frames.
  - Using a frozen ResNet reduces computation and leverages learned features from ImageNet.
- **Why GRU?**
  - GRU learns how features evolve over time without requiring as much memory as LSTM.
  - Ideal for modeling short sequences efficiently.

### 2. ResNet2D + LSTM
- **Why LSTM over GRU here?**
  - LSTM has a more expressive memory, allowing it to capture long-term dependencies.
  - Suitable when the key event (like a goal) happens after a sequence of earlier contextual movements.

### 3. ResNet3D (R3D)
- **Why ResNet3D?**
  - Applies 3D convolutions over spatiotemporal volumes — so it captures motion and appearance together.
  - No need for a separate temporal model like GRU/LSTM because temporal learning is built into the convolutions.
- **When is this better?**
  - More powerful for short clips with well-localized events, as it models motion directly.

---

## 🧪 Supervised Learning Setup
- Each video clip is labeled with a class (e.g., Goal, Foul, etc.)
- We train our models in a supervised fashion to minimize classification loss using **CrossEntropyLoss**.

---

## 🧼 Preprocessing Pipeline
1. **Video Collection**: Full soccer matches are cut into short clips of 1–2 seconds.
2. **Frame Extraction**: Extract frames at a fixed frame rate.
3. **Resize & Normalize**: Resize to 224x224 and normalize for ResNet input.
4. **Clip Stacking**:
   - For ResNet2D+GRU/LSTM: Frames are passed independently.
   - For ResNet3D: Clips are stacked into 3D tensors.

---

## 🧠 Training Details
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing
- **Loss**: Cross-Entropy
- **Early Stopping**: Based on validation accuracy
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix

---

## 📊 Evaluation
We evaluated each model on a held-out validation set using:
- Confusion matrix to analyze misclassifications
- Precision/Recall/F1-score to measure model balance

---

## 🎨 Visuals
We included diagrams in our report to illustrate:
- Flow of input through each model
- Comparison of architectures
- Differences between 2D+GRU/LSTM and 3D CNNs

---

## 💡 Key Learnings
- 3D CNNs capture motion elegantly but are more computationally expensive.
- Using pretrained ResNet2D significantly reduces training time and improves performance.
- LSTM handles delayed events better than GRU in certain sequences.

---

## 📂 Folder Structure
```
├── data/
│   ├── raw_videos/
│   └── processed_clips/
├── models/
│   ├── resnet2d_gru.py
│   ├── resnet2d_lstm.py
│   └── resnet3d.py
├── train.py
├── evaluate.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## 🚀 Future Work
- Explore transformer-based video models like TimeSformer
- Integrate audio cues and commentary for multi-modal classification
- Deploy as a live highlight detector using stream input

---

## 🙌 Contributors
- Jenish Kothari
- Vidith Agarwal
- Meet Katrodiya
- Yash Phalle

---

## 📬 Feedback
Feel free to open issues or reach out with suggestions or collaboration ideas!
