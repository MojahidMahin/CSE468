"""
GPU Image Cache for Medical Image Captioning

Pre-loads and caches preprocessed medical images in GPU VRAM for efficient batch processing.
Implements memory monitoring and graceful fallback strategies.
"""

import torch
import gc
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Optional


class GPUImageCache:
    """Cache preprocessed images in GPU VRAM for batch inference."""

    def __init__(self, processor, device: str, dtype: torch.dtype, vram_limit_gb: float = 9.0):
        """
        Initialize GPU image cache.

        Args:
            processor: HuggingFace processor (e.g., AutoProcessor)
            device: Device to cache to ('cuda' or 'cpu')
            dtype: Data type for cached tensors (torch.float16 or torch.float32)
            vram_limit_gb: VRAM limit in GB before stopping cache
        """
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.vram_limit_gb = vram_limit_gb
        self.cache = []  # List of dicts with cached tensors
        self.metadata = []  # List of dicts with image metadata

        print(f"\nInitialized GPU Image Cache")
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")
        print(f"  VRAM Limit: {vram_limit_gb} GB")

    def _get_vram_usage(self) -> float:
        """Get current GPU VRAM usage in GB."""
        if self.device != 'cuda':
            return 0.0

        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1e9

    def _get_tensor_memory(self, tensor: torch.Tensor) -> float:
        """Get memory size of tensor in MB."""
        if tensor is None:
            return 0.0
        return tensor.element_size() * tensor.nelement() / 1e6

    def _create_messages(self, image):
        """Create message format for processor."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this medical image in detail."}
                ]
            }
        ]

    def _process_image(self, messages):
        """Process image through model processor."""
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)

        # Create inputs
        inputs = self.processor(
            text=[text],
            images=[image_inputs],
            videos=None,
            padding=True,
            return_tensors='pt'
        )

        return inputs

    def preprocess_and_cache(self, dataset, max_images: int = 1000) -> int:
        """
        Pre-process and store images in GPU VRAM.

        Args:
            dataset: HuggingFace dataset with images
            max_images: Maximum number of images to cache

        Returns:
            Number of images successfully cached
        """
        print(f"\nPre-loading images to GPU VRAM...")
        print(f"Target: {min(max_images, len(dataset))} images")
        print("-" * 80)

        images_cached = 0
        vram_peak = self._get_vram_usage()

        for idx, sample in enumerate(tqdm(dataset, total=min(max_images, len(dataset)), desc="Pre-caching images")):
            if idx >= max_images:
                break

            try:
                # Check VRAM before caching
                current_vram = self._get_vram_usage()
                if current_vram > self.vram_limit_gb:
                    print(f"\nVRAM limit reached ({current_vram:.2f} GB > {self.vram_limit_gb} GB)")
                    print(f"Stopped at {idx} images")
                    break

                # Extract data
                image = sample['image']
                image_id = sample['image_id']
                original_caption = sample.get('caption', '')

                # Preprocess image
                messages = self._create_messages(image)
                inputs = self._process_image(messages)

                # Store on GPU with specified dtype
                cached_item = {
                    'pixel_values': inputs['pixel_values'].to(self.device, dtype=self.dtype),
                    'image_grid_thw': inputs['image_grid_thw'].to(self.device),
                    'input_ids': inputs['input_ids'].to(self.device),
                    'attention_mask': inputs['attention_mask'].to(self.device)
                }

                self.cache.append(cached_item)
                self.metadata.append({
                    'image_id': image_id,
                    'original_caption': original_caption,
                    'cache_index': idx
                })

                images_cached += 1
                vram_peak = max(vram_peak, current_vram)

            except Exception as e:
                print(f"Error caching image {idx}: {e}")
                continue

        print("-" * 80)
        print(f"\nCache Summary:")
        print(f"  Images cached: {images_cached}")
        print(f"  VRAM peak usage: {vram_peak:.2f} GB / {self.vram_limit_gb} GB")
        print(f"  Cache size: ~{images_cached * 2.9:.2f} GB (estimated)")

        return images_cached

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Retrieve batch of preprocessed images from cache.

        Args:
            indices: List of cache indices to retrieve

        Returns:
            Dictionary with batched tensors (already on GPU)
        """
        if not indices:
            raise ValueError("indices list cannot be empty")

        if max(indices) >= len(self.cache):
            raise IndexError(f"Index {max(indices)} out of cache range {len(self.cache)}")

        # Collect tensors for batch
        pixel_values_list = []
        image_grid_thw_list = []
        input_ids_list = []
        attention_mask_list = []

        for idx in indices:
            cached = self.cache[idx]
            pixel_values_list.append(cached['pixel_values'])
            image_grid_thw_list.append(cached['image_grid_thw'])
            input_ids_list.append(cached['input_ids'])
            attention_mask_list.append(cached['attention_mask'])

        # Concatenate into batched tensors
        batch = {
            'pixel_values': torch.cat(pixel_values_list, dim=0),
            'image_grid_thw': torch.cat(image_grid_thw_list, dim=0),
            'input_ids': torch.cat(input_ids_list, dim=0),
            'attention_mask': torch.cat(attention_mask_list, dim=0)
        }

        return batch

    def get_metadata(self, indices: List[int]) -> List[Dict]:
        """Get metadata for batch indices."""
        return [self.metadata[idx] for idx in indices]

    def clear_cache(self):
        """Clear all cached data from GPU."""
        self.cache.clear()
        self.metadata.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print("Cache cleared from GPU")

    def get_cache_size(self) -> int:
        """Get number of images in cache."""
        return len(self.cache)

    def get_vram_status(self) -> Dict:
        """Get VRAM usage statistics."""
        current = self._get_vram_usage()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9 if self.device == 'cuda' else 0
        available = total - current

        return {
            'current_gb': current,
            'total_gb': total,
            'available_gb': available,
            'percent_used': (current / total * 100) if total > 0 else 0
        }
