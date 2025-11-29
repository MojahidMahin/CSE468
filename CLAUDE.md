# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multi-VLM Image Captioning Framework** that compares multiple lightweight Vision-Language Models on the COCO dataset. The project is optimized for RTX 5080 (16GB VRAM) and processes images to generate captions using various transformer models.

## Architecture

The project uses a modular architecture with the following key components:

### Core Components

1. **Configuration Management** (`Config` class in `cse468_vlm_processing.py`)
   - Centralized settings for annotations path, image directories, results directory
   - Device detection (GPU/CPU) and VRAM information
   - Configurable number of images to process

2. **Model Classes** (Modular VLM implementations)
   - `QwenVLMCaptioner` - Qwen2-VL-2B (5-6 GB VRAM) - default enabled
   - Additional model classes available (commented) for MobileVLM, LLaVA, Phi-3, InternVL2, SmolVLM2, DeepSeek-VL
   - Each model class handles its own initialization, caption generation, and memory cleanup

3. **Data Pipeline**
   - COCO dataset integration using `pycocotools`
   - Image loading with PIL
   - Batch processing with progress tracking via tqdm
   - Checkpoint saving every 50 images for fault tolerance

4. **Output System**
   - CSV-based result storage
   - Per-model results files: `results/results_{model_name}.csv`
   - Combined comparison file: `results/all_models_comparison.csv`
   - Periodic checkpoint files during processing

### Directory Structure

```
/home/vortex/CSE 468 AFE/Project/
├── cse468_vlm_processing.py          # Main script (recommended for PyCharm)
├── cse468_project_task_1_refactored.ipynb  # Jupyter notebook version
├── cse468_project_multi_vlm_complete.ipynb # Full multi-model notebook
├── annotations/                      # COCO annotation files
├── coco_images/                      # Downloaded COCO images
├── results/                          # Output CSV files and checkpoints
├── requirements.txt                  # Python dependencies
└── README_REFACTORED.md              # Quick start guide
```

## Running the Code

### Prerequisites

Install dependencies from an activated virtual environment:
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Option 1: Python Script (PyCharm)

```bash
python cse468_vlm_processing.py
```

This is the recommended approach for development in PyCharm as it doesn't require Jupyter dependencies.

### Option 2: Jupyter Notebook

```bash
jupyter notebook cse468_project_task_1_refactored.ipynb
```

Or use VS Code with Jupyter extension.

### Option 3: Full Multi-Model Comparison

```bash
jupyter notebook cse468_project_multi_vlm_complete.ipynb
```

Includes all 7 VLM models (can selectively enable/disable models).

## Configuration

Modify configuration in `cse468_vlm_processing.py` (the `Config` class):

```python
class Config:
    ANNOTATIONS_PATH = '/home/vortex/CSE 468 AFE/Project/annotations'
    IMAGES_DIR = 'coco_images'
    RESULTS_DIR = 'results'
    NUM_IMAGES = 200  # Change number of images to process
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Key settings:
- `NUM_IMAGES`: How many images to process (default: 200)
- `IMAGES_DIR`: Where to store/load COCO images
- `RESULTS_DIR`: Where to save results CSV files
- `ANNOTATIONS_PATH`: Path to COCO annotations JSON files

## Key Dependencies

The project requires:
- **PyTorch** (2.9.1+) - Deep learning framework
- **Transformers** (4.57.1+) - HuggingFace model loading
- **PIL/Pillow** - Image processing
- **pandas** - CSV output handling
- **pycocotools** - COCO dataset API
- **tqdm** - Progress bars
- **qwen-vl-utils** - Qwen model-specific utilities

All dependencies are specified in `requirements.txt`.

## Data Flow

1. **Initialization**: `Config.validate()` checks paths and sets up directories
2. **Annotation Loading**: COCO class loads annotation JSON files
3. **Image Processing Loop**:
   - Load image from COCO dataset
   - Pass image through selected VLM model
   - Record caption, processing time, and image metadata
   - Save checkpoint every 50 images
4. **Memory Management**: Models are unloaded after processing to free VRAM for next model
5. **Output**: Results saved to CSV with columns: `image_id, model_name, caption, processing_time_sec, image_width, image_height, timestamp`

## GPU Memory Management

The project implements automatic memory management:
- Models load with `device_map="auto"` for distributed loading across available GPU memory
- Float16 precision used to reduce VRAM usage
- Explicit model unloading after processing each image batch
- `gc.collect()` and `torch.cuda.empty_cache()` called between models

To check available VRAM:
```python
import torch
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Model Configurations

### Enabled Models
- **Qwen2-VL-2B** (5-6 GB) - Default, good balance of speed and quality

### Available But Commented Out
- **MobileVLM-V2** (6-8 GB) - Optimized for mobile/edge
- **LLaVA-1.5** (14-16 GB) - Better reasoning (tight on 16GB VRAM)
- **Phi-3-Vision** (8-10 GB) - Microsoft's efficient model
- **InternVL2** (4-6 GB) - OpenGVLab's compact model
- **SmolVLM2** (5.2 GB) - HuggingFace ultra-efficient, supports video
- **DeepSeek-VL** (4-5 GB) - Efficient alternative

To enable additional models, find their commented class definitions in the script and uncomment, then add to `MODELS_CONFIG` list.

## Working with Notebooks

When editing Jupyter notebooks:
- The notebook cells correspond to logical steps in the Python script
- Configuration changes in the "Config" cell apply to that notebook run
- Results are generated in the same CSV format as the script
- Each cell can be run independently for debugging

## Output Format

Results are saved as CSV files with the following columns:
```
image_id | model_name | caption | processing_time_sec | image_width | image_height | timestamp
```

Files generated:
- `results/results_{ModelName}.csv` - Per-model detailed results
- `results/all_models_comparison.csv` - Combined results from all processed models
- `results/checkpoint_*.csv` - Intermediate checkpoints (every 50 images)

## Checkpoint System

The processing pipeline saves checkpoints every 50 images. If interrupted:
1. Check `results/checkpoint_*.csv` for latest checkpoint
2. The pipeline can resume from checkpoint on next run (implementation depends on script version)
3. Final results combine all processed images into the main output CSV

## Performance Considerations

- **Processing Time**: Varies by model and GPU. Qwen2-VL-2B typically processes 1-2 images per second on RTX 5080
- **VRAM Scaling**: Single-model processing requires 5-16 GB depending on chosen model
- **Multi-Model Comparison**: Process sequentially with automatic unloading between models to stay within 16GB VRAM
- **Batch Size**: Currently set to process one image at a time; can be modified for batch processing

## Development Notes

- The project uses `trust_remote_code=True` when loading models from HuggingFace to support custom model implementations
- Image downloading is handled by the COCO API; the script assumes annotations are present locally
- All timestamps use UTC via Python's `datetime` module
- Progress tracking uses tqdm for console visualization

## Testing and Validation

To test with a small subset:
1. Set `Config.NUM_IMAGES = 10` in the Config class
2. Run the script to verify setup works
3. Check `results/results_Qwen2-VL-2B.csv` for output format
4. Increase `NUM_IMAGES` as needed

## Common Issues and Debugging

### CUDA Out of Memory (OOM)
- Reduce `NUM_IMAGES` for testing
- Enable a smaller model (InternVL2 at 4-6 GB)
- Ensure no other CUDA-using processes are running

### Missing Model Weights
- First run downloads model weights from HuggingFace (~2-5 GB per model)
- Requires stable internet connection
- Models cache locally in `.cache/huggingface/` directory

### Annotation Loading Errors
- Verify `annotations/` directory exists
- Ensure COCO annotation JSON files are present
- Check that paths in `Config` point to correct locations
