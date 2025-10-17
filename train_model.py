import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

# ------------------------------
# 1. Device (GPU if available)
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# 2. Paths
# ------------------------------
train_dir = r"C:\Users\Deven Aggarwal\Desktop\Plantapp\dataset\train"
val_dir = r"C:\Users\Deven Aggarwal\Desktop\Plantapp\dataset\val"

# ------------------------------
# 3. Classes
# ------------------------------
class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
num_classes = len(class_names)
print("Classes found:", class_names)

# ------------------------------
# 4. Hyperparameters
# ------------------------------
num_epochs = 2        # Quick test
batch_size = 32
learning_rate = 0.001

# ------------------------------
# 5. Transforms
# ------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ------------------------------
# 6. Datasets
# ------------------------------
full_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
full_val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

# ------------------------------
# 7. Subset for fast testing
# ------------------------------
# Use smaller number of images for quick training
train_indices = list(range(len(full_train_dataset)))
random.shuffle(train_indices)
train_dataset = Subset(full_train_dataset, train_indices[:1000])  # first 1000 images

val_indices = list(range(len(full_val_dataset)))
random.shuffle(val_indices)
val_dataset = Subset(full_val_dataset, val_indices[:500])  # first 500 images

# ------------------------------
# 8. DataLoaders
# ------------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# 9. Model, Loss, Optimizer
# ------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------------------
# 10. Training Loop
# ------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # ------------------------------
    # Validation
    # ------------------------------
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_acc = val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.4f}\n")

# ------------------------------
# 11. Save model
# ------------------------------
torch.save(model.state_dict(), "plant_model.pth")
print("Training complete and model saved as plant_model.pth")
