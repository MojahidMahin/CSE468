"""
Medical Image Captioning for ROCOv2-radiology Dataset
Generates captions for radiology images using Vision-Language Models
optimized for medical imaging tasks.
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

warnings.filterwarnings('ignore')


class MedicalConfig:
    """Configuration for medical image captioning."""

    # Dataset paths
    DATASET_PATH = '/home/vortex/CSE 468 AFE/Datasets/ROCOv2-radiology'
    RESULTS_DIR = '/home/vortex/CSE 468 AFE/Project/results_medical'

    # Processing settings
    NUM_IMAGES = 50  # Change for full dataset (59962 train, 9904 val, 9927 test)
    SPLIT = 'train'    # 'train', 'validation', or 'test'

    # Device and model settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model selection - optimized for medical imaging
    MODEL_NAME = 'Qwen/Qwen2-VL-2B-Instruct'  # Supports medical image understanding

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

            # Get image info from messages - process_vision_info returns a tuple of (images, videos)
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


def process_medical_images():
    """Main pipeline for medical image captioning."""

    # Validate configuration
    MedicalConfig.validate()

    # Load dataset
    processor = ROCOv2Processor()
    dataset = processor.load_split()
    batch = processor.get_batch()

    # Initialize model
    captioner = MedicalVLMCaptioner(MedicalConfig.MODEL_NAME)

    # Process images
    results = []

    print(f"\nProcessing {len(batch)} medical images...")
    print("-" * 80)

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
                'split': MedicalConfig.SPLIT
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

    # Save final results
    results_df = pd.DataFrame(results)
    output_file = save_results(results_df)

    # Print summary
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Total images processed: {len(results)}")
    print(f"Average processing time: {results_df['processing_time_sec'].mean():.2f} seconds")
    print(f"Results saved to: {output_file}")

    # Cleanup
    captioner.cleanup()

    return results_df


if __name__ == '__main__':
    results_df = process_medical_images()
