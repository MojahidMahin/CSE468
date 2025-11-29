"""
Multi-VLM Image Captioning Framework
Comparing multiple lightweight vision-language models on COCO dataset
Optimized for RTX 5080 (16GB VRAM)
"""

import os
import sys
import json
import time
import random
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

# Try to import required libraries, with helpful error messages
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install transformers qwen-vl-utils")
    sys.exit(1)


class Config:
    """Configuration for the VLM processing pipeline"""
    ANNOTATIONS_PATH = '/home/vortex/CSE 468 AFE/Project/annotations'
    IMAGES_DIR = 'coco_images'
    RESULTS_DIR = 'results'
    NUM_IMAGES = 200
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def validate(cls):
        """Validate configuration paths exist"""
        if not os.path.exists(cls.ANNOTATIONS_PATH):
            print(f"Error: Annotations path not found: {cls.ANNOTATIONS_PATH}")
            sys.exit(1)

        os.makedirs(cls.IMAGES_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

        print(f"Setup complete.")
        print(f"  Annotations: {cls.ANNOTATIONS_PATH}")
        print(f"  Images: {cls.IMAGES_DIR}")
        print(f"  Results: {cls.RESULTS_DIR}")
        print(f"  Device: {cls.DEVICE}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class QwenVLMCaptioner:
    """
    Qwen2-VL-2B Vision-Language Model captioning
    Alibaba's efficient model optimized for vision-language tasks
    """

    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    DISPLAY_NAME = "Qwen2-VL-2B"
    VRAM_ESTIMATE = "5-6 GB"

    def __init__(self):
        """Initialize model and processor"""
        print(f"\nLoading {self.DISPLAY_NAME}...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"Loaded {self.DISPLAY_NAME}")
            print(f"Estimated VRAM: {self.VRAM_ESTIMATE}")
        except Exception as e:
            print(f"Failed to load {self.DISPLAY_NAME}: {e}")
            raise

    def generate_caption(self, image: Image.Image) -> str:
        """
        Generate caption for a single image

        Args:
            image: PIL Image object

        Returns:
            Caption string
        """
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            device = Config.DEVICE
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256)

            full_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract assistant response
            if "assistant" in full_output:
                caption = full_output.split("assistant")[-1].strip()
            else:
                caption = full_output

            return caption
        except Exception as e:
            return f"Error: {str(e)[:100]}"

    def unload(self):
        """Free GPU memory"""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        gc.collect()


def load_coco_dataset() -> Tuple[COCO, List[str]]:
    """
    Load COCO annotations and prepare image list

    Returns:
        Tuple of (COCO object, list of image filenames)
    """
    print(f"\nLoading COCO annotations...")
    coco = COCO(os.path.join(Config.ANNOTATIONS_PATH, 'captions_val2017.json'))

    random.seed(42)  # For reproducibility
    all_img_ids = coco.getImgIds()
    selected_img_ids = random.sample(all_img_ids, Config.NUM_IMAGES)

    print(f"Selected {len(selected_img_ids)} images for processing")

    # Ensure images directory exists and download missing images
    os.makedirs(Config.IMAGES_DIR, exist_ok=True)

    image_files = []
    missing_count = 0

    for img_id in tqdm(selected_img_ids, desc="Preparing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(Config.IMAGES_DIR, img_info['file_name'])

        if not os.path.exists(img_path):
            try:
                import requests
                img_url = img_info['coco_url']
                img_data = requests.get(img_url).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                print(f"Failed to download {img_info['file_name']}: {e}")
                missing_count += 1
                continue

        image_files.append(img_info['file_name'])

    print(f"Ready with {len(image_files)}/{Config.NUM_IMAGES} images")

    return coco, image_files


def process_images_with_model(
    model: QwenVLMCaptioner,
    image_files: List[str]
) -> List[Dict]:
    """
    Process all images with a VLM model

    Args:
        model: Captioner model instance
        image_files: List of image filenames

    Returns:
        List of result dictionaries
    """
    results = []
    start_time = time.time()
    model_name = model.DISPLAY_NAME

    print(f"\n{'='*80}")
    print(f"Processing with: {model_name}")
    print(f"{'='*80}")
    print(f"Expected time: ~{len(image_files) * 2.5 / 60:.1f} minutes")

    for idx, img_file in enumerate(tqdm(image_files, desc=f"Processing with {model_name}")):
        try:
            img_path = os.path.join(Config.IMAGES_DIR, img_file)
            image = Image.open(img_path).convert('RGB')

            img_id = img_file.replace('.jpg', '')
            img_size = image.size

            start_inference = time.time()
            caption = model.generate_caption(image)
            inference_time = time.time() - start_inference

            result = {
                'image_id': img_id,
                'model_name': model_name,
                'caption': caption,
                'processing_time_sec': round(inference_time, 2),
                'image_width': img_size[0],
                'image_height': img_size[1],
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

            # Save checkpoint every 50 images
            if (idx + 1) % 50 == 0:
                checkpoint_path = os.path.join(
                    Config.RESULTS_DIR,
                    f"checkpoint_{model_name}_{idx+1}.csv"
                )
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                elapsed = (time.time() - start_time) / 60
                tqdm.write(f"Checkpoint {idx+1}/{len(image_files)}: {elapsed:.1f}m elapsed")

        except Exception as e:
            tqdm.write(f"Error processing {img_file}: {str(e)[:50]}")
            continue

    # Save model results
    result_path = os.path.join(Config.RESULTS_DIR, f"results_{model_name}.csv")
    pd.DataFrame(results).to_csv(result_path, index=False)

    elapsed_time = (time.time() - start_time) / 60
    successful = len([r for r in results if not r['caption'].startswith('Error')])

    print(f"\nCompleted {model_name}:")
    print(f"  Processed: {len(results)}/{len(image_files)} images")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Time: {elapsed_time:.1f} minutes")
    if len(results) > 0:
        print(f"  Avg per image: {elapsed_time * 60 / len(results):.1f}s")
    print(f"  Saved to: {result_path}")

    return results


def analyze_results(all_results: List[Dict]):
    """
    Analyze and display results statistics

    Args:
        all_results: List of all result dictionaries
    """
    if not all_results:
        print("No results to analyze")
        return

    df = pd.DataFrame(all_results)

    # Save combined results
    combined_path = os.path.join(Config.RESULTS_DIR, 'all_models_comparison.csv')
    df.to_csv(combined_path, index=False)

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    for model_name in df['model_name'].unique():
        model_data = df[df['model_name'] == model_name]
        caption_lengths = model_data['caption'].str.len()

        print(f"\n{model_name}:")
        print(f"  Total images: {len(model_data)}")
        print(f"  Successful: {len([c for c in model_data['caption'] if not c.startswith('Error')])}:")
        print(f"  Avg caption length: {caption_lengths.mean():.0f} characters")
        print(f"  Caption range: {caption_lengths.min()}-{caption_lengths.max()}")
        print(f"  Avg inference time: {model_data['processing_time_sec'].mean():.2f}s")

    print(f"\nAll results saved to: {combined_path}")


def main():
    """Main processing pipeline"""
    print("\n" + "="*80)
    print("Multi-VLM Image Captioning Framework")
    print("="*80)

    # Setup
    Config.validate()

    # Load dataset
    coco, image_files = load_coco_dataset()

    if not image_files:
        print("No images to process!")
        return

    # Initialize and process with Qwen model
    all_results = []

    try:
        qwen_model = QwenVLMCaptioner()
        results = process_images_with_model(qwen_model, image_files)
        all_results.extend(results)
        qwen_model.unload()
        time.sleep(2)  # Give GPU time to free memory
    except Exception as e:
        print(f"Error processing with Qwen: {e}")

    # Analyze results
    analyze_results(all_results)

    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
