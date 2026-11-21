# 🐱 vs 🐶 Cat vs Dog Image Classifier in PyTorch

A clean, production-grade PyTorch conversion of the original TensorFlow/Keras Google Colab project.

---

## 📋 Project Summary & Architecture Comparison

| Feature / Step | Original TensorFlow / Keras Code | Converted PyTorch Code |
| :--- | :--- | :--- |
| **Framework** | TensorFlow 2.x / Keras | PyTorch (`torch`, `torchvision`) |
| **Device Acceleration** | Automatic / Default | Dynamic CUDA detection (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) |
| **Data Preprocessing** | `image_size=(224, 224)`, `images / 255.0` | `transforms.Resize((224, 224))`, `transforms.ToTensor()` (auto-normalizes to `[0.0, 1.0]`) |
| **Dataset Loading** | `tf.keras.utils.image_dataset_from_directory` | `torchvision.datasets.ImageFolder` + `DataLoader` |
| **Input Shape** | `(224, 224, 3)` (NHWC) | `(3, 224, 224)` (NCHW) |
| **Flatten Layer** | `Flatten()` $\to$ $150,528$ features | `nn.Flatten()` $\to$ $3 \times 224 \times 224 = 150,528$ features |
| **Hidden Layer 1** | `Dense(228, activation='relu')` | `nn.Linear(150528, 228)` + `nn.ReLU()` |
| **Regularization** | `Dropout(0.2)` | `nn.Dropout(p=0.2)` |
| **Hidden Layer 2** | `Dense(64, activation='relu')` | `nn.Linear(228, 64)` + `nn.ReLU()` |
| **Output Layer** | `Dense(1, activation='sigmoid')` | `nn.Linear(64, 1)` + `nn.Sigmoid()` |
| **Optimizer** | `Adam` (lr=0.001) | `torch.optim.Adam(model.parameters(), lr=0.001)` |
| **Loss Function** | `binary_crossentropy` | `nn.BCELoss()` |
| **Epochs & Batch Size**| 40 Epochs, Batch Size = 2 | 40 Epochs, Batch Size = 2 |
| **Model Saving** | `model.save('cat_vs_dog.keras')` | `torch.save(model.state_dict(), 'cat_vs_dog.pth')` & `cat_vs_dog_full.pth` |
| **Model Loading** | `load_model('cat_vs_dog.keras')` | `loaded_model.load_state_dict(torch.load('cat_vs_dog.pth'))` |
| **Inference Decision** | `prediction > 0.5` $\to$ Dog, else Cat | `probability > 0.5` $\to$ Dog, else Cat |
| **Training Visualization**| Missing / Unused | Loss & Accuracy curves plotted via `matplotlib.pyplot` |
| **Image Display** | Terminal print only | `matplotlib.pyplot` inline image viewer with predicted labels & confidence scores |

---

## 🚀 How to Open and Run in Google Colab

1. **Download the Notebook**:
   - Location: `S:\Antigravity\artifacts\Cat_vs_Dog_PyTorch_Colab.ipynb`
2. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/).
   - Click **File** $\to$ **Upload notebook** $\to$ Select `Cat_vs_Dog_PyTorch_Colab.ipynb`.
3. **Select GPU Accelerator (Optional but Recommended)**:
   - Click **Runtime** $\to$ **Change runtime type** $\to$ Choose **T4 GPU** $\to$ Click **Save**.
4. **Upload Dataset**:
   - Upload your `Project CS (2).zip` file to Colab's file browser in the left sidebar.
5. **Run All Cells**:
   - Click **Runtime** $\to$ **Run all** (or press `Ctrl + F9`).
