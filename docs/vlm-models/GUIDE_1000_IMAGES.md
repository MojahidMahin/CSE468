# Processing 1000 Images - User Guide

## What's Been Optimized

Your `cse468_project_task_1_refactored.ipynb` has been updated for processing 1000 images:

### Changes Made:

1. ✅ **NUM_IMAGES = 1000** - Process all 1000 images you've already downloaded
2. ✅ **Reduced tokens** - Max 128 tokens instead of 256 (faster inference)
3. ✅ **Shorter prompts** - "Describe briefly" instead of detailed (faster)
4. ✅ **More frequent checkpoints** - Every 100 images instead of 50
5. ✅ **Better progress tracking** - Shows remaining time estimate
6. ✅ **Optimized for RTX 5080** - Memory-efficient inference

## Timeline

| Metric | Value |
|--------|-------|
| Total Images | 1000 |
| Avg Time/Image | 2-3 seconds |
| Checkpoint Interval | Every 100 images |
| **Total Expected Time** | **33-50 minutes** |
| **Expected Hours** | **0.5-0.8 hours** |

## Running the Notebook

### Step 1: Restart Kernel
Since the previous run was stuck, restart to clear memory:
- **Kernel → Restart & Clear Output**

### Step 2: Run Cells in Order

1. **Cell 0** - Setup and imports (installs packages)
   - Time: 2 minutes

2. **Cell 2** - Configuration (should show 1000 images now)
   - Time: <1 minute

3. **Cell 4** - Load COCO dataset
   - Uses your already-downloaded 1000 images
   - Time: 1-2 minutes

4. **Cell 6** - Load Qwen model
   - First run: 2-3 minutes (downloads weights)
   - Subsequent runs: 30 seconds
   - Time: 2-3 minutes

5. **Cell 8** - Configuration (skip, already configured)

6. **Cell 9** - **MAIN PROCESSING**
   - **This is where the 33-50 minutes happens**
   - Shows progress with checkpoints
   - Can't be interrupted without losing progress (save checkpoints!)
   - Time: **33-50 minutes**

7. **Cell 11** - Results analysis
   - Time: <1 minute

### Step 3: Monitor Progress

While cell 9 runs, you'll see:

```
Processing with: Qwen2-VL-2B
================================================================================
Estimated time: 33-50 minutes
Checkpoints will be saved every 100 images
================================================================================

Processing with Qwen2-VL-2B: 10%|█         | 100/1000 [03:20<30:15, 1.82s/it]
✓ Checkpoint 100/1000: 5.6m elapsed, ~50m remaining
```

### Checkpoints

Every 100 images, a checkpoint file is saved:
```
results/
├── checkpoint_Qwen2-VL-2B_100.csv
├── checkpoint_Qwen2-VL-2B_200.csv
├── checkpoint_Qwen2-VL-2B_300.csv
├── ...
└── checkpoint_Qwen2-VL-2B_1000.csv
```

If interrupted, you can resume from the last checkpoint.

## Output Files

### Final Results
```
results/
├── results_Qwen2-VL-2B.csv          # All 1000 image results
└── all_models_comparison.csv        # Combined (same as above for single model)
```

### CSV Format
```
image_id,model_name,caption,processing_time_sec,image_width,image_height,timestamp
000000391895,Qwen2-VL-2B,"A dog sitting...",2.34,640,480,2025-11-18T10:30:45
000000554625,Qwen2-VL-2B,"An urban street...",2.10,800,600,2025-11-18T10:31:20
...
```

## Performance Optimization Tips

### 1. Use Idle System
- Close other applications
- Avoid using computer during processing
- Let GPU focus on inference

### 2. Monitor GPU
In another terminal, watch GPU usage:
```bash
watch -n 1 nvidia-smi
```

Expected:
- GPU Memory: ~6 GB (Qwen model)
- GPU Utilization: 80-100%
- Temperature: 60-75°C

### 3. Network
- Don't stream/download during processing
- First load downloads model once (already cached after)

### 4. If It Gets Stuck
- Don't kill the process during processing
- Only interrupt between checkpoints
- Check last checkpoint file to see progress

## If Something Goes Wrong

### Issue: "CUDA out of memory"
- Your RTX 5080 has 16GB, Qwen uses ~6GB - should be fine
- If it happens, try restarting notebook and reducing NUM_IMAGES to 500

### Issue: Model loading takes too long
- First run downloads ~4.5GB model weights - normal
- Should be faster on subsequent runs
- Check internet connection if very slow

### Issue: Processing is very slow (>5s per image)
- Normal is 2-3s, RTX 5080 should handle it
- Close other applications
- Check GPU temperature (shouldn't exceed 85°C)

### Issue: Interrupted mid-processing
- Check checkpoint files - last one completed
- You can manually resume from there by modifying the loop

## Resume From Checkpoint

If interrupted at checkpoint 500, you can:

1. Look at `checkpoint_Qwen2-VL-2B_500.csv` - shows what's been processed
2. Modify the processing loop to skip first 500
3. Or just re-run from start (it overwrites results, which is fine)

## What's Happening

### Qwen2-VL-2B Model
- 2 billion parameters
- Uses float16 (half precision) for speed
- Runs on RTX 5080 with ~6GB VRAM
- ~2-3 seconds per image on your GPU

### Process Flow
1. Load image from disk
2. Resize/process to model input size
3. Run inference (forward pass)
4. Generate 128 tokens max
5. Extract caption text
6. Save to results
7. Clear GPU cache between images

## After Processing

Once complete, you'll have:

✅ **1000 image captions** in CSV format
✅ **Checkpoints** saved every 100 images
✅ **Processing times** for each image
✅ **Image dimensions** recorded
✅ **Timestamps** for tracking

You can then:
- Load CSV into Python/Pandas for analysis
- Create visualizations
- Filter by caption length
- Compare with other models
- Export for further processing

## Next Steps (Optional)

After getting results with Qwen (1 model), you can:

1. **Add more models** using `cse468_project_multi_vlm_complete.ipynb`
   - Run SmolVLM2-2.2B (smaller, faster)
   - Run Phi-3-Vision (better quality)
   - Compare all results

2. **Analyze captions**
   - Caption length distribution
   - Processing time per model
   - Quality comparison

3. **Export results**
   - CSV to Excel
   - Create reports
   - Build visualizations

---

## Expected Result

After ~40 minutes:
```
================================================================================
COMPLETED Qwen2-VL-2B
================================================================================
Processed: 1000/1000 images
Successful: 1000/1000 (100.0%)
Total time: 40.5 minutes (0.7 hours)
Avg per image: 2.43s
Results saved to: results/results_Qwen2-VL-2B.csv
================================================================================
```

Good luck with your 1000 image processing! 🚀
