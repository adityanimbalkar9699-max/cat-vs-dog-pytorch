"""
Cat vs Dog Image Classifier in PyTorch
Converted 1:1 from TensorFlow/Keras Google Colab Project
"""

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

def extract_dataset(zip_candidates=None, extract_to='/Project_SC'):
    if zip_candidates is None:
        zip_candidates = [
            '/Project CS (2).zip',
            '/content/Project CS (2).zip',
            '/Project CS.zip',
            '/content/Project CS.zip',
            'Project CS (2).zip',
            'Project CS.zip'
        ]
    
    zip_path = None
    for p in zip_candidates:
        if os.path.exists(p):
            zip_path = p
            break
            
    if zip_path and os.path.exists(zip_path):
        print(f"Extracting {zip_path} -> {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
        return extract_to
    elif os.path.exists(os.path.join(extract_to, 'Project CS', 'Train')):
        print(f"Dataset already exists at: {extract_to}/Project CS/Train")
        return extract_to
    elif os.path.exists('/content/Project_SC/Project CS/Train'):
        return '/content/Project_SC'
    else:
        print("Notice: Zip archive not found at standard locations.")
        return extract_to

class CatDogClassifier(nn.Module):
    """
    Exact PyTorch equivalent of the Keras Sequential Model:
    - Input: (3, 224, 224)
    - Flatten: 3 * 224 * 224 = 150,528
    - Dense(228, relu)
    - Dropout(0.2)
    - Dense(64, relu)
    - Dense(1, sigmoid)
    """
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(3 * 224 * 224, 228),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(228, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

def train_and_evaluate(train_dir, epochs=40, batch_size=2, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Image transformations: Resize to (224, 224) and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if os.name != 'nt' else 0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Loaded {len(train_dataset)} images from {len(train_dataset.classes)} classes: {train_dataset.classes}")
    print(f"Class indices: {train_dataset.class_to_idx}")

    model = CatDogClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'loss': [], 'accuracy': []}

    print(f"\n--- Starting Training ({epochs} Epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f}")

    # Evaluation
    model.eval()
    eval_loss = 0.0
    eval_correct = 0
    eval_total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * images.size(0)
            preds = (outputs >= 0.5).float()
            eval_correct += (preds == labels).sum().item()
            eval_total += labels.size(0)

    print(f"\nFinal Evaluation Results:")
    print(f"Loss: {eval_loss / eval_total:.4f}")
    print(f"Accuracy: {eval_correct / eval_total:.4f}")

    return model, history, transform

def predict_test_images(model, test_dir, transform, device):
    model.eval()
    test_images = []
    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append(os.path.join(root, file))

    if not test_images:
        print(f"No test images found in {test_dir}")
        return

    print(f"\nFound {len(test_images)} test image(s):")
    for img_path in test_images:
        raw_img = Image.open(img_path).convert('RGB')
        tensor_img = transform(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(tensor_img).item()

        label = "dog" if prob > 0.5 else "cat"
        confidence = prob if prob > 0.5 else (1.0 - prob)
        print(f"  {os.path.basename(img_path)} -> {label.upper()} (Score: {prob:.4f}, Confidence: {confidence:.2%})")

if __name__ == '__main__':
    extract_dir = extract_dataset()
    train_path = os.path.join(extract_dir, 'Project CS', 'Train')
    test_path = os.path.join(extract_dir, 'Project CS', 'Test')
    
    if os.path.exists(train_path):
        trained_model, hist, xform = train_and_evaluate(train_path)
        # Save model
        torch.save(trained_model.state_dict(), 'cat_vs_dog.pth')
        print("Model saved to cat_vs_dog.pth")
        predict_test_images(trained_model, test_path, xform, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        print(f"Path not found: {train_path}. Please place your dataset in the correct folder.")
