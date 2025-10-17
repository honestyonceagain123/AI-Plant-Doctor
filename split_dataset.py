import os
import shutil
import random

# ✅ Step 1: Path to your main dataset folder
source_dir = r"C:\Users\Deven Aggarwal\Desktop\PlantApp\archive\PlantVillage"

# ✅ Step 2: Base directory (where new folders will be created)
base_dir = r"C:\Users\Deven Aggarwal\Desktop\PlantApp"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# ✅ Step 3: Create target folders if they don’t exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# ✅ Step 4: Define split ratio (80% train, 20% validation)
split_ratio = 0.8

# ✅ Step 5: Loop through each plant class folder
for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    # Filter image files only
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(images) == 0:
        print(f"⚠️ No images found in class '{cls}', skipping...")
        continue

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_files = images[:split_index]
    val_files = images[split_index:]

    # Create subfolders in train/ and val/
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Copy files
    for f in train_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(train_dir, cls, f))
    for f in val_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(val_dir, cls, f))

print("✅ Dataset split into train and validation successfully!")
