"""
Medical Image Captioning with Integrated Evaluation Metrics
Generates captions and evaluates them against original captions.
"""

import os
import gc
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings

from medical_image_metrics import MetricsAggregator

warnings.filterwarnings('ignore')


class MedicalConfig:
    """Configuration for medical image captioning."""

    # Dataset paths
    DATASET_PATH = '/home/vortex/CSE 468 AFE/Datasets/ROCOv2-radiology'
    RESULTS_DIR = '/home/vortex/CSE 468 AFE/Project/results_medical'

    # Processing settings
    NUM_IMAGES = 1000  # Change for full dataset (59962 train, 9904 val, 9927 test)
    SPLIT = 'train'    # 'train', 'validation', or 'test'

    # Device and model settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model selection - optimized for medical imaging
    MODEL_NAME = 'Qwen/Qwen2-VL-2B-Instruct'  # Supports medical image understanding

    # Evaluation settings
    COMPUTE_BERT_SCORE = False  # Set to True for semantic similarity (slower)

    # GPU optimization settings
    ENABLE_GPU_CACHE = True      # Enable VRAM pre-loading of images
    BATCH_SIZE = 8               # Images per batch (4, 8, or 16)
    GPU_CACHE_VRAM_LIMIT = 9.0   # Safety limit in GB before stopping cache

    @classmethod
    def validate(cls):
        """Validate configuration and create necessary directories."""
        # Check dataset exists
        if not os.path.exists(cls.DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {cls.DATASET_PATH}")

        # Create results directory
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

        # Print device info
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Total VRAM: {total_mem:.1f} GB")
        else:
            print("Device: CPU")


class MedicalVLMCaptioner:
    """Medical image captioning using Vision-Language Models."""

    def __init__(self, model_name):
        """Initialize medical VLM model."""
        print(f"\nLoading model: {model_name}")
        self.model_name = model_name
        self.device = MedicalConfig.DEVICE

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Use the correct model class for Qwen2-VL
        from transformers import Qwen2VLForConditionalGeneration
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map='auto',
            dtype=MedicalConfig.DTYPE,
            trust_remote_code=True
        )
        self.model.eval()

    def generate_caption(self, image):
        """
        Generate caption for a single medical image.

        Args:
            image: PIL Image object

        Returns:
            caption: Generated text caption
            processing_time: Time taken in seconds
        """
        import time
        start_time = time.time()

        try:
            # Prepare input for medical image understanding
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this medical image in detail."}
                    ]
                }
            ]

            # Process text and image
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Get image info from messages
            image_inputs, video_inputs = process_vision_info(messages)

            # Create inputs
            inputs = self.processor(
                text=[text],
                images=[image_inputs],
                videos=None,
                padding=True,
                return_tensors='pt'
            )

            inputs = inputs.to(self.device)

            # Generate with medical-specific parameters
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            # Decode output - get only the new tokens generated
            prompt_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, prompt_length:]
            response = self.processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            processing_time = time.time() - start_time
            return response.strip(), processing_time

        except Exception as e:
            print(f"Error generating caption: {e}")
            processing_time = time.time() - start_time
            return f"Error: {str(e)}", processing_time

    def generate_caption_batch(self, batch_inputs):
        """
        Generate captions for a batch of images (GPU pre-loaded).

        Args:
            batch_inputs: Dictionary with batched tensors already on GPU
                - pixel_values: [batch_size, ...]
                - image_grid_thw: [batch_size, 3]
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]

        Returns:
            captions: List of generated caption strings
            proc_times: List of per-image processing times
        """
        import time
        start_time = time.time()

        try:
            batch_size = batch_inputs['input_ids'].shape[0]

            # Generate captions for entire batch
            with torch.no_grad():
                output_ids = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            # Decode each caption separately
            captions = []
            prompt_length = batch_inputs['input_ids'].shape[1]

            for i in range(batch_size):
                generated_ids = output_ids[i, prompt_length:]
                caption = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                captions.append(caption.strip())

            processing_time = time.time() - start_time
            per_image_time = processing_time / batch_size if batch_size > 0 else 0

            return captions, [per_image_time] * batch_size

        except Exception as e:
            print(f"Error in batch caption generation: {e}")
            batch_size = batch_inputs['input_ids'].shape[0]
            return [f"Error: {str(e)}"] * batch_size, [0.0] * batch_size

    def cleanup(self):
        """Clean up model memory."""
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


class ROCOv2Processor:
    """Handle ROCOv2 dataset loading and processing."""

    def __init__(self):
        """Initialize dataset processor."""
        self.dataset = None
        self.split = MedicalConfig.SPLIT

    def load_split(self):
        """Load specified split from ROCOv2 dataset."""
        print(f"\nLoading {self.split} split from ROCOv2...")
        self.dataset = load_dataset(
            'parquet',
            data_files={
                self.split: os.path.join(
                    MedicalConfig.DATASET_PATH,
                    'data',
                    f'{self.split}-*.parquet'
                )
            }
        )[self.split]

        print(f"Loaded {len(self.dataset)} images from {self.split} split")
        return self.dataset

    def get_batch(self, num_images=None):
        """Get batch of images for processing."""
        if num_images is None:
            num_images = MedicalConfig.NUM_IMAGES

        num_images = min(num_images, len(self.dataset))
        return self.dataset.select(range(num_images))


def save_results(results_df, suffix=''):
    """Save results to CSV."""
    output_file = os.path.join(
        MedicalConfig.RESULTS_DIR,
        f'medical_captions_{MedicalConfig.SPLIT}{suffix}.csv'
    )
    results_df.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")
    return output_file


def process_medical_images_with_eval():
    """Main pipeline for medical image captioning with integrated evaluation."""

    # Validate configuration
    MedicalConfig.validate()

    # Load dataset
    processor = ROCOv2Processor()
    dataset = processor.load_split()
    batch = processor.get_batch()

    # Initialize model
    captioner = MedicalVLMCaptioner(MedicalConfig.MODEL_NAME)

    # Initialize metrics aggregator
    metrics_aggregator = MetricsAggregator()

    # Initialize GPU cache if enabled
    gpu_cache = None
    if MedicalConfig.ENABLE_GPU_CACHE:
        from gpu_image_cache import GPUImageCache

        print("\n" + "=" * 100)
        print("GPU Image Cache Enabled - Pre-loading images to VRAM")
        print("=" * 100)

        gpu_cache = GPUImageCache(
            captioner.processor,
            MedicalConfig.DEVICE,
            MedicalConfig.DTYPE,
            vram_limit_gb=MedicalConfig.GPU_CACHE_VRAM_LIMIT
        )
        num_cached = gpu_cache.preprocess_and_cache(
            batch,
            max_images=MedicalConfig.NUM_IMAGES
        )

        print(f"\nSuccessfully cached {num_cached} images")
        vram_status = gpu_cache.get_vram_status()
        print(f"VRAM Status: {vram_status['current_gb']:.2f} GB / {vram_status['total_gb']:.2f} GB ({vram_status['percent_used']:.1f}%)")

    # Process images
    results = []

    if gpu_cache is not None:
        # Batched processing with GPU cache
        num_batches = (len(batch) + MedicalConfig.BATCH_SIZE - 1) // MedicalConfig.BATCH_SIZE
        print(f"\nProcessing {len(batch)} medical images in batches of {MedicalConfig.BATCH_SIZE}...")
        print("-" * 100)

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            try:
                start_idx = batch_idx * MedicalConfig.BATCH_SIZE
                end_idx = min(start_idx + MedicalConfig.BATCH_SIZE, len(batch))
                indices = list(range(start_idx, end_idx))

                # Get batch from GPU cache
                batch_inputs = gpu_cache.get_batch(indices)

                # Generate captions for entire batch
                captions, proc_times = captioner.generate_caption_batch(batch_inputs)

                # Get metadata for batch
                batch_metadata = gpu_cache.get_metadata(indices)

                # Record results
                for i, idx in enumerate(indices):
                    metadata = batch_metadata[i]
                    result = {
                        'image_id': metadata['image_id'],
                        'original_caption': metadata['original_caption'],
                        'generated_caption': captions[i],
                        'processing_time_sec': proc_times[i],
                        'model': MedicalConfig.MODEL_NAME,
                        'timestamp': datetime.utcnow().isoformat(),
                        'split': MedicalConfig.SPLIT,
                        'batch_size': MedicalConfig.BATCH_SIZE
                    }
                    results.append(result)

                # Save checkpoint every 25 images
                if (end_idx) % 25 == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_file = os.path.join(
                        MedicalConfig.RESULTS_DIR,
                        f'checkpoint_{end_idx}.csv'
                    )
                    checkpoint_df.to_csv(checkpoint_file, index=False)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        # Cleanup GPU cache before evaluation
        gpu_cache.clear_cache()

    else:
        # Sequential processing (original behavior)
        print(f"\nProcessing {len(batch)} medical images sequentially...")
        print("-" * 100)

        for idx, sample in enumerate(tqdm(batch, desc="Generating captions")):
            try:
                image = sample['image']
                image_id = sample['image_id']
                original_caption = sample.get('caption', '')

                # Generate caption
                caption, proc_time = captioner.generate_caption(image)

                # Record result
                result = {
                    'image_id': image_id,
                    'original_caption': original_caption,
                    'generated_caption': caption,
                    'processing_time_sec': proc_time,
                    'model': MedicalConfig.MODEL_NAME,
                    'timestamp': datetime.utcnow().isoformat(),
                    'split': MedicalConfig.SPLIT,
                    'batch_size': 1
                }
                results.append(result)

                # Save checkpoint every 25 images
                if (idx + 1) % 25 == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_file = os.path.join(
                        MedicalConfig.RESULTS_DIR,
                        f'checkpoint_{idx + 1}.csv'
                    )
                    checkpoint_df.to_csv(checkpoint_file, index=False)

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

    # Save results
    results_df = pd.DataFrame(results)
    output_file = save_results(results_df)

    # Cleanup model
    captioner.cleanup()

    # Evaluate captions
    print("\n" + "=" * 100)
    print("Evaluating Generated Captions")
    print("=" * 100)

    # Filter out error captions for evaluation
    valid_results = results_df[~results_df['generated_caption'].str.startswith('Error')].copy()

    if len(valid_results) > 0:
        detailed_metrics, aggregate_stats = metrics_aggregator.evaluate_batch(
            valid_results['original_caption'].tolist(),
            valid_results['generated_caption'].tolist(),
            include_bert=MedicalConfig.COMPUTE_BERT_SCORE
        )

        # Merge metrics with results
        results_with_metrics = valid_results.copy()
        for col in detailed_metrics.columns:
            if col != 'image_index':
                results_with_metrics[col] = detailed_metrics[col].values

        # Save evaluation results
        metrics_output_file = os.path.join(
            MedicalConfig.RESULTS_DIR,
            f'medical_captions_{MedicalConfig.SPLIT}_with_metrics.csv'
        )
        results_with_metrics.to_csv(metrics_output_file, index=False)
        print(f"Saved results with metrics to: {metrics_output_file}")

        # Save aggregate statistics
        metrics_aggregator.save_results(
            detailed_metrics,
            aggregate_stats,
            MedicalConfig.RESULTS_DIR,
            f'medical_{MedicalConfig.SPLIT}'
        )

        # Print summary
        metrics_aggregator.print_summary(aggregate_stats, f"Medical Image Captioning Evaluation - {MedicalConfig.SPLIT} split")
    else:
        print("No valid captions to evaluate (all captions had errors)")

    # Print final summary
    print("\n" + "=" * 100)
    print("Processing Complete!")
    print("=" * 100)
    print(f"Total images processed: {len(results)}")
    print(f"Valid captions for evaluation: {len(valid_results)}")
    print(f"Average processing time: {results_df['processing_time_sec'].mean():.2f} seconds")
    print(f"Results saved to: {output_file}")

    return results_df


if __name__ == '__main__':
    results_df = process_medical_images_with_eval()
