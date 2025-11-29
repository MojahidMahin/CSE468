# Medical Image Captioning - Quick Start

## 5-Minute Setup

### 1. Run the Script
```bash
cd "/home/vortex/CSE 468 AFE/Project"
python medical_image_captioning.py
```

### 2. Check Results
```bash
ls -lh results_medical/
head -5 results_medical/medical_captions_train.csv
```

### 3. View Results
```python
import pandas as pd
df = pd.read_csv('results_medical/medical_captions_train.csv')
print(df[['image_id', 'original_caption', 'generated_caption']].head())
```

## Configuration Quick Reference

Edit `MedicalConfig` class in `medical_image_captioning.py`:

```python
NUM_IMAGES = 50            # Change number of images to process
SPLIT = 'train'            # Choose: 'train', 'validation', 'test'
MODEL_NAME = 'Qwen/Qwen2-VL-2B-Instruct'  # Current model
```

## Processing Times

| Images | Time | Notes |
|--------|------|-------|
| 10 | 1 min | Quick test |
| 50 | 5 min | Standard test |
| 100 | 10 min | Medium batch |
| 500 | 50 min | Large batch |
| 1000 | 1.5 hrs | Full batch |
| 59,962 | 100+ hrs | Full training set |

## Output File Locations

- **Results**: `results_medical/medical_captions_{split}.csv`
- **Checkpoints**: `results_medical/checkpoint_*.csv` (saved every 25 images)
- **Documentation**: `MEDICAL_CAPTIONING_GUIDE.md`

## Sample CSV Structure

```
image_id | original_caption | generated_caption | processing_time_sec | split
ROCOv2_2023_train_000001 | Head CT demonstrating left parotiditis. | This medical image is a computed tomography (CT) scan... | 6.66 | train
```

## Troubleshooting

### Issue: Script won't start
```bash
# Install missing dependencies
pip install -r requirements.txt
pip install datasets pyarrow
```

### Issue: CUDA Out of Memory
```python
# Reduce batch size in medical_image_captioning.py
NUM_IMAGES = 10  # Instead of 50
```

### Issue: No CUDA/GPU available
```python
# Script auto-falls back to CPU (slower)
# Force CPU: DEVICE = 'cpu'
```

## Resources

- **Full Guide**: See `MEDICAL_CAPTIONING_GUIDE.md`
- **Dataset**: `/home/vortex/CSE 468 AFE/Datasets/ROCOv2-radiology/`
- **Model**: Qwen2-VL-2B-Instruct (HuggingFace)
- **Results Directory**: `/home/vortex/CSE 468 AFE/Project/results_medical/`

## Key Statistics (50 Image Test)

✅ **Total Processed**: 50 images
✅ **Average Speed**: 5.88 seconds/image
✅ **Total Time**: ~5 minutes
✅ **VRAM Used**: 6.5 GB (RTX 5080)
✅ **Success Rate**: 100%

---

**Version**: 1.0 | **Tested**: RTX 5080 | **Status**: Working ✓
