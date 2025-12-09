"""
Data Loading Module for Radiology Dataset

Handles loading ground truth captions and images from the dataset.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from config import EvaluationConfig


class RadiologydataLoader:
    """Loader for radiology dataset with ground truth captions"""

    def __init__(self, config: EvaluationConfig = EvaluationConfig):
        self.config = config
        self.ground_truth_data = []
        self.load_ground_truth()

    def load_ground_truth(self):
        """Load ground truth data from JSONL file"""
        print(f"Loading ground truth from: {self.config.GROUND_TRUTH_FILE}")

        with open(self.config.GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.ground_truth_data.append(data)

        print(f"✓ Loaded {len(self.ground_truth_data)} ground truth entries")

    def get_dataset_info(self) -> Dict:
        """Get dataset statistics"""
        total_images = len(self.ground_truth_data)
        modalities = {}

        for entry in self.ground_truth_data:
            modality = entry.get('metadata', {}).get('modality', 'unknown')
            modalities[modality] = modalities.get(modality, 0) + 1

        return {
            'total_images': total_images,
            'modality_distribution': modalities,
            'sample_ids': [entry['id'] for entry in self.ground_truth_data[:5]]
        }

    def get_sample(self, idx: int) -> Dict:
        """
        Get a single sample with image and ground truth caption

        Returns:
            Dict with keys: id, image_path, image, caption, reasoning
        """
        if idx >= len(self.ground_truth_data):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self.ground_truth_data)})")

        entry = self.ground_truth_data[idx]
        image_path = self.config.DATA_DIR / entry['image_path']

        # Load image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')

        return {
            'id': entry['id'],
            'image_path': str(image_path),
            'image': image,
            'caption': entry['caption'],
            'reasoning': entry.get('reasoning', ''),
            'metadata': entry.get('metadata', {})
        }

    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict]:
        """Get a batch of samples"""
        batch = []
        for i in range(start_idx, min(start_idx + batch_size, len(self.ground_truth_data))):
            try:
                batch.append(self.get_sample(i))
            except Exception as e:
                print(f"Warning: Failed to load sample {i}: {e}")
                continue
        return batch

    def get_samples_by_limit(self, num_samples: int) -> List[Dict]:
        """Get first N samples for evaluation"""
        if num_samples > len(self.ground_truth_data):
            print(f"Warning: Requested {num_samples} samples but only {len(self.ground_truth_data)} available")
            num_samples = len(self.ground_truth_data)

        samples = []
        for i in range(num_samples):
            try:
                samples.append(self.get_sample(i))
            except Exception as e:
                print(f"Warning: Failed to load sample {i}: {e}")
                continue

        return samples

    def __len__(self):
        """Return total number of samples"""
        return len(self.ground_truth_data)

    def __getitem__(self, idx):
        """Support indexing"""
        return self.get_sample(idx)


if __name__ == "__main__":
    # Test data loader module
    print("Testing Data Loader Module...\n")

    try:
        # Initialize loader
        loader = RadiologydataLoader()

        # Get dataset info
        info = loader.get_dataset_info()
        print(f"\nDataset Information:")
        print(f"  Total images: {info['total_images']}")
        print(f"  Modality distribution: {info['modality_distribution']}")
        print(f"  Sample IDs: {info['sample_ids']}")

        # Test loading a single sample
        print(f"\nTesting single sample loading...")
        sample = loader.get_sample(0)
        print(f"  Sample ID: {sample['id']}")
        print(f"  Image path: {sample['image_path']}")
        print(f"  Image size: {sample['image'].size}")
        print(f"  Caption: {sample['caption'][:100]}...")

        # Test batch loading
        print(f"\nTesting batch loading (5 samples)...")
        batch = loader.get_batch(0, 5)
        print(f"  Loaded {len(batch)} samples in batch")

        # Test limited samples
        print(f"\nTesting limited sample loading (10 samples)...")
        samples = loader.get_samples_by_limit(10)
        print(f"  Loaded {len(samples)} samples")

        print("\n✓ Data loader module test passed!")

    except Exception as e:
        print(f"\n✗ Data loader module test failed: {e}")
        raise
