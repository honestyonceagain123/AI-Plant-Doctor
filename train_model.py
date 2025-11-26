import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_model():
    # ✅ Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    if device.type == "cuda":
        print(f"🚀 GPU in use: {torch.cuda.get_device_name(0)}")

    data_dir = os.path.join(os.getcwd(), "dataset")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # 🔁 Data transforms
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    # 📦 Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms["val"])

    # 🔢 DataLoaders — use pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    num_classes = len(train_dataset.classes)
    print(f"🌱 Total classes: {num_classes}")

    # 🧠 Model setup
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 🧩 Load previous best model if exists (resume from last checkpoint)
    if os.path.exists("best_plant_model.pth"):
        model.load_state_dict(torch.load("best_plant_model.pth", map_location=device))
        print("✅ Loaded previous best model — resuming or using trained weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc = 0
    epochs = 20

    for epoch in range(epochs):
        print(f"\n🧩 Epoch [{epoch+1}/{epochs}]")
        model.train()
        running_loss, running_corrects = 0.0, 0

        loop = tqdm(train_loader, desc=f"🧠 Training Epoch {epoch+1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_acc = running_corrects.double() / len(train_dataset)
        train_loss = running_loss / len(train_dataset)

        # 🧪 Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_dataset)
        val_loss = val_loss / len(val_dataset)
        scheduler.step(val_acc)

        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"✅ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 🔥 Log GPU memory usage
        if device.type == "cuda":
            mem_used = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"🧠 GPU Memory — Used: {mem_used:.1f} MB | Reserved: {mem_reserved:.1f} MB")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_plant_model.pth")
            print(f"💾 New best model saved! (Accuracy: {best_acc:.4f})")

    print(f"\n🎉 Training complete! Best Validation Accuracy: {best_acc:.4f}")
    print("✅ Model saved as best_plant_model.pth")

# 🧩 Required fix for Windows multiprocessing
if __name__ == "__main__":
    torch.cuda.empty_cache()  # ensure GPU memory is clear before training
    train_model()
