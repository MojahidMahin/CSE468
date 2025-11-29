# Multi-VLM Image Captioning Framework

Refactored notebook and scripts for comparing multiple lightweight Vision-Language Models on COCO dataset, optimized for RTX 5080 (16GB VRAM).

## Files

### 1. **cse468_vlm_processing.py** (Recommended for PyCharm)
- Standalone Python script that runs directly in PyCharm
- No Jupyter dependency needed
- Same functionality as notebook but in script format
- Run with: `python cse468_vlm_processing.py`

### 2. **cse468_project_task_1_refactored.ipynb**
- Jupyter notebook version
- Can be run in Jupyter Lab/Notebook or VS Code with Jupyter extension
- Same code as Python script but organized in cells
- Better for interactive exploration

## Quick Start

### Option A: Using Python Script (PyCharm)
```bash
cd "/home/vortex/CSE 468 AFE/Project"
python cse468_vlm_processing.py
```

### Option B: Using Jupyter Notebook
```bash
cd "/home/vortex/CSE 468 AFE/Project"
jupyter notebook cse468_project_task_1_refactored.ipynb
```

## Configuration

Edit `cse468_vlm_processing.py` or the Config cell in the notebook to change:
- `ANNOTATIONS_PATH`: Path to COCO annotations (default: already set)
- `IMAGES_DIR`: Where to store downloaded images
- `RESULTS_DIR`: Where to save results
- `NUM_IMAGES`: Number of images to process (default: 200)

## Models

Currently configured with **Qwen2-VL-2B** (5-6 GB VRAM).

To add more models, uncomment sections in the code for:
- MobileVLM-V2 (3B) - 6-8 GB
- LLaVA-1.5 (7B) - 14-16 GB
- Phi-3-Vision (4.2B) - 8-10 GB
- InternVL2 (2B) - 4-6 GB
- SmolVLM2 (2.2B) - 5.2 GB
- DeepSeek-VL-1.3B - 4-5 GB

## Output Format

Results are saved as CSV files with structure:
```
image_id, model_name, caption, processing_time_sec, image_width, image_height, timestamp
```

Files saved to `results/`:
- `results_Qwen2-VL-2B.csv` - Per-model results
- `all_models_comparison.csv` - Combined results from all models
- `checkpoint_*.csv` - Periodic checkpoints during processing

## Key Features

✅ All old Gemini code kept but commented out
✅ No Google Colab dependencies
✅ Uses local filesystem only
✅ Sequential model processing with memory management
✅ Progress tracking with checkpoints
✅ Human-like code comments throughout

## Requirements

```bash
pip install transformers torch torchvision Pillow
pip install accelerate bitsandbytes qwen-vl-utils
pip install pycocotools pandas tqdm
```

## GPU Memory Management

Models automatically unload after processing to free VRAM for the next model.
Progress is saved every 50 images in case of interruption.

## Support for Adding New Models

See the commented model classes in the Python script to add:
- MobileVLMModel
- LLaVAModel
- PhiVisionModel
- InternVL2Model
- SmolVLM2Model
- DeepSeekVLModel

Just uncomment and enable in MODELS_CONFIG.
