import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

zip_path = 'Project CS (2).zip'
if not os.path.exists(zip_path):
    zip_path = '/content/Project CS (2).zip'
if not os.path.exists(zip_path):
    zip_path = '/Project CS (2).zip'

extract_path = '/Project_SC'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Folder connected and extracted to:", extract_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = '/Project_SC/Project CS/Train'
if not os.path.exists(train_dir) and os.path.exists('/content/Project_SC/Project CS/Train'):
    train_dir = '/content/Project_SC/Project CS/Train'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(224 * 224 * 3, 228),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(228, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 40
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
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
        
        total_loss += loss.item() * len(labels)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
        
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f}")

model.eval()
val_loss = 0.0
val_correct = 0
val_total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * len(labels)
        preds = (outputs >= 0.5).float()
        val_correct += (preds == labels).sum().item()
        val_total += len(labels)

loss = val_loss / val_total
accuracy = val_correct / val_total
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

torch.save(model.state_dict(), 'cat_vs_dog.pth')
