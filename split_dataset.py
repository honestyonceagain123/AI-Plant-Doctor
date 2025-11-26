import os
import shutil
import random
from tqdm import tqdm

# ---------- Paths ----------
BASE_DIR = r"C:\Users\Deven Aggarwal\Desktop\Plantapp"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
COMBINED_DIR = os.path.join(DATASET_DIR, "combined")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# ---------- Split ratios ----------
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ---------- Allowed image extensions ----------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# ---------- Step 1: Combine datasets ----------
def combine_all_datasets():
    print("🌿 Combining all datasets...")
    if os.path.exists(COMBINED_DIR):
        shutil.rmtree(COMBINED_DIR)
    os.makedirs(COMBINED_DIR)

    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path) or folder in ["train", "val", "test", "combined"]:
            continue

        print(f"📂 Adding dataset: {folder}")
        # Look for nested folders (like Train/Test)
        subdirs = []
        for sub in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub)
            if os.path.isdir(sub_path):
                subdirs.append(sub_path)
        if not subdirs:
            subdirs = [folder_path]  # No nesting, just use root

        # Copy all images from each subdirectory
        for subdir in subdirs:
            for class_name in os.listdir(subdir):
                class_path = os.path.join(subdir, class_name)
                if not os.path.isdir(class_path):
                    continue

                dest_class_dir = os.path.join(COMBINED_DIR, class_name)
                os.makedirs(dest_class_dir, exist_ok=True)

                for img_name in os.listdir(class_path):
                    src_path = os.path.join(class_path, img_name)
                    if not os.path.isfile(src_path) or not img_name.lower().endswith(IMAGE_EXTS):
                        continue  # skip folders or non-images

                    dest_path = os.path.join(dest_class_dir, f"{folder}_{img_name}")
                    try:
                        shutil.copy2(src_path, dest_path)
                    except PermissionError:
                        continue  # skip any locked or unreadable files

    print("✅ All datasets merged into:", COMBINED_DIR)

# ---------- Step 2: Split dataset ----------
def split_dataset():
    print("✂️ Splitting dataset into train, val, and test...")
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    classes = os.listdir(COMBINED_DIR)
    print(f"🌱 Found {len(classes)} classes: {classes[:10]}{'...' if len(classes) > 10 else ''}")

    for class_name in tqdm(classes):
        class_dir = os.path.join(COMBINED_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTS)]
        if len(images) < 5:
            print(f"⚠️ Skipping '{class_name}' (too few samples: {len(images)})")
            continue
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_SPLIT)
        n_val = int(n_total * VAL_SPLIT)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR: images[n_train:n_train + n_val],
            TEST_DIR: images[n_train + n_val:],
        }

        for split_dir, split_images in splits.items():
            dest_dir = os.path.join(split_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_name in split_images:
                src_path = os.path.join(class_dir, img_name)
                dest_path = os.path.join(dest_dir, img_name)
                shutil.copy2(src_path, dest_path)

    print("✅ Dataset split complete!")
    print(f"Train dir: {TRAIN_DIR}\nVal dir: {VAL_DIR}\nTest dir: {TEST_DIR}")

if __name__ == "__main__":
    combine_all_datasets()
    split_dataset()
    print("🎉 Merged + Split dataset ready for training!")
