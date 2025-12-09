#!/bin/bash
# Full Evaluation Script for 500+ Images with All 4 Models
# This script runs the complete zero-shot evaluation on radiology images

echo "========================================"
echo "VLM Zero-Shot Evaluation - Full Run"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - Number of images: 100"
echo "  - Models: Qwen2-VL-2B, Qwen3-VL-4B, Qwen2.5-VL-7B"
echo "  - Metrics: BLEU, ROUGE, METEOR, ChrF, BERTScore, Perplexity"
echo ""
echo "Note: This will take approximately 30-45 minutes depending on your GPU"
echo ""
read -p "Press Enter to start evaluation or Ctrl+C to cancel..."

cd "/home/vortex/CSE 468 AFE/Project/Evaluation/codes"

echo ""
echo "========================================"
echo "Running Model 1/3: Qwen2-VL-2B"
echo "========================================"
#python run_evaluation.py --num-images 100 --models Qwen2-VL-2B

echo ""
echo "========================================"
echo "Running Model 2/3: Qwen3-VL-4B"
echo "========================================"
#python run_evaluation.py --num-images 100 --models Qwen3-VL-4B

echo ""
echo "========================================"
echo "Running Model 3/3: Qwen2.5-VL-7B"
echo "========================================"
python run_evaluation.py --num-images 100 --models Qwen2.5-VL-7B
echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved in: /home/vortex/CSE 468 AFE/Project/Evaluation/results/"
echo ""
echo "To view results:"
echo "  - Model outputs: results/model_outputs/"
echo "  - Metrics: results/metrics/evaluation_summary.csv"
echo "  - Visualizations: results/visualizations/"


# /bin/bash "/home/vortex/CSE 468 AFE/Project/Evaluation/codes/run_full_evaluation_500.sh"
