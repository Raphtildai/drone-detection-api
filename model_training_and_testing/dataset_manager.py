# dataset_manager.py

import os, random, shutil, zipfile, urllib.request
from pathlib import Path

# ==================== DATASET MANAGER ====================
class DatasetManager:
    def __init__(self, config):
        self.config = config

    def prepare_dataset(self):
        """Prepare and split dataset"""
        if (self.config.PROCESSED_DIR / "train").exists():
            print("✅ Dataset already exists")
            return True

        repo_dir = self._download_and_extract()
        if not repo_dir:
            return False

        return self._process_dataset(repo_dir)

    def _download_and_extract(self):
        """Download dataset"""
        extracted_dir = self.config.RAW_DIR / "DroneAudioDataset-master"
        if extracted_dir.exists():
            print("✅ Dataset already extracted")
            return extracted_dir

        zip_path = self.config.RAW_DIR / "drone_repo.zip"
        print("📥 Downloading dataset...")
        urllib.request.urlretrieve(self.config.GITHUB_ZIP_URL, str(zip_path))

        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(self.config.RAW_DIR)
        zip_path.unlink()

        print("✅ Dataset ready")
        return extracted_dir

    def _process_dataset(self, repo_dir):
        """Process dataset with balancing"""
        binary_dir = repo_dir / "Binary_Drone_Audio"
        if not binary_dir.exists():
            print("❌ Binary_Drone_Audio not found")
            return False

        classes_found = [item.name for item in binary_dir.iterdir() if item.is_dir()]
        if not classes_found:
            print("❌ No class folders found")
            return False

        return self._process_class_directories(binary_dir, classes_found)

    def _process_class_directories(self, binary_dir, classes_found):
        """Process with class balancing"""
        class_mapping = {
            "yes_drone": "drone", "unknown": "non_drone",
            "Drone": "drone", "noDrone": "non_drone",
        }

        # Collect files
        all_files = {"drone": [], "non_drone": []}
        for class_dir in classes_found:
            target_class = class_mapping.get(class_dir.lower(), "non_drone")
            src_folder = binary_dir / class_dir
            files = list(src_folder.glob("*.wav"))
            all_files[target_class].extend(files)

        drone_files = all_files["drone"]
        non_drone_files = all_files["non_drone"]

        print(f"📊 Raw - Drone: {len(drone_files)}, Non-drone: {len(non_drone_files)}")

        # Balance dataset
        if len(non_drone_files) > len(drone_files):
            non_drone_files = random.sample(non_drone_files, min(len(drone_files) * 2, len(non_drone_files)))

        print(f"⚖️ Balanced - Drone: {len(drone_files)}, Non-drone: {len(non_drone_files)}")

        # Create splits
        for target_class, files in [("drone", drone_files), ("non_drone", non_drone_files)]:
            self._create_splits(target_class, files)

        total_files = sum(1 for _ in self.config.PROCESSED_DIR.rglob("*.wav"))
        print(f"✅ Processed dataset: {total_files} files")
        return total_files > 0

    def _create_splits(self, target_class, files):
        """Create train/val/test splits"""
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val

        splits = {"train": files[:n_train], "val": files[n_train:n_train+n_val], "test": files[n_train+n_val:]}

        for split, file_list in splits.items():
            dest = self.config.PROCESSED_DIR / split / target_class
            dest.mkdir(parents=True, exist_ok=True)
            for f in file_list:
                dst = dest / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
