"""
Configuration Module for VLM Zero-Shot Evaluation on Radiology Dataset

This module contains all configuration settings for the evaluation pipeline.
"""

import os
import torch
from pathlib import Path


class EvaluationConfig:
    """Central configuration for VLM evaluation pipeline"""

    # ========== Paths ==========
    BASE_DIR = Path("/home/vortex/CSE 468 AFE/Project/Evaluation")
    DATA_DIR = BASE_DIR / "Reason-datasets"
    IMAGES_DIR = DATA_DIR / "images"
    GROUND_TRUTH_FILE = DATA_DIR / "data.jsonl"

    # Output directories
    OUTPUT_DIR = BASE_DIR / "results"
    OUTPUTS_DIR = OUTPUT_DIR / "model_outputs"
    METRICS_DIR = OUTPUT_DIR / "metrics"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

    # ========== Model Configuration ==========
    MODELS_CONFIG = [
        {
            "name": "Qwen2-VL-2B",
            "model_id": "Qwen/Qwen2-VL-2B-Instruct",
            "vram_gb": 6,
            "enabled": True
        },
        {
            "name": "Qwen3-VL-4B",
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "vram_gb": 8,
            "enabled": True
        },
        {
            "name": "Qwen2.5-VL-7B",
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "vram_gb": 12,
            "enabled": True
        },
        {
            "name": "Phi3-Vision",
            "model_id": "microsoft/Phi-3-vision-128k-instruct",
            "vram_gb": 10,
            "enabled": True
        },
        {
            "name": "InternVL2-2B",
            "model_id": "OpenGVLab/InternVL2-2B",
            "vram_gb": 6,
            "enabled": True
        },
        {
            "name": "SmolVLM2",
            "model_id": "HuggingFaceTB/SmolVLM2-Instruct",
            "vram_gb": 5.2,
            "enabled": True
        }
    ]

    # ========== Processing Configuration ==========
    NUM_IMAGES = 500  # Minimum required images (can be increased to 1999)
    BATCH_SIZE = 1  # Process one image at a time for VRAM efficiency
    CHECKPOINT_INTERVAL = 50  # Save checkpoint every N images

    # ========== Device Configuration ==========
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FP16 = True  # Use float16 for memory efficiency

    # ========== Evaluation Metrics ==========
    METRICS = [
        "BLEU",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "METEOR",
        "ChrF",
        "BERTScore",
        "Perplexity"
    ]

    # ========== Prompts for Zero-Shot Inference ==========
    # Medical radiology-specific prompt
    ZERO_SHOT_PROMPT = (
        # "Describe this medical image in detail. "
        "Provide a concise radiology report finding for this medical image. "
        "Describe only the key clinical observations in 1-2 sentences."
        "Include observations about anatomical structures, "
        "any visible abnormalities, and key clinical findings."
    )

    @classmethod
    def validate(cls):
        """Validate that all required paths exist and create output directories"""
        # Check input paths
        if not cls.DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found: {cls.DATA_DIR}")
        if not cls.IMAGES_DIR.exists():
            raise FileNotFoundError(f"Images directory not found: {cls.IMAGES_DIR}")
        if not cls.GROUND_TRUTH_FILE.exists():
            raise FileNotFoundError(f"Ground truth file not found: {cls.GROUND_TRUTH_FILE}")

        # Create output directories
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

        print("✓ Configuration validated successfully")
        print(f"  - Ground truth: {cls.GROUND_TRUTH_FILE}")
        print(f"  - Images directory: {cls.IMAGES_DIR}")
        print(f"  - Output directory: {cls.OUTPUT_DIR}")
        print(f"  - Device: {cls.DEVICE}")
        if cls.DEVICE == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  - Available VRAM: {total_vram:.1f} GB")

    @classmethod
    def get_enabled_models(cls):
        """Get list of enabled models"""
        return [m for m in cls.MODELS_CONFIG if m["enabled"]]

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("EVALUATION CONFIGURATION")
        print("="*60)
        print(f"Number of images to process: {cls.NUM_IMAGES}")
        print(f"Checkpoint interval: {cls.CHECKPOINT_INTERVAL}")
        print(f"Device: {cls.DEVICE}")
        print(f"FP16 enabled: {cls.USE_FP16}")
        print(f"\nEnabled Models ({len(cls.get_enabled_models())} total):")
        for model in cls.get_enabled_models():
            print(f"  - {model['name']} ({model['vram_gb']} GB VRAM)")
        print(f"\nMetrics to compute:")
        for metric in cls.METRICS:
            print(f"  - {metric}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test configuration module
    print("Testing Configuration Module...\n")

    # Validate configuration
    try:
        EvaluationConfig.validate()
        EvaluationConfig.print_config()
        print("\n✓ Configuration module test passed!")
    except Exception as e:
        print(f"\n✗ Configuration module test failed: {e}")
        raise
