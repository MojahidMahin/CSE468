# Medical Image Captioning Evaluation Guide

This guide explains how to generate and use evaluation metrics for your medical image captioning system.

## Overview

The evaluation system provides the following metrics:

### Standard Metrics (NLTK-based)
- **BLEU Score** - Measures n-gram overlap with reference captions
- **METEOR** - Considers synonyms and word stems
- **ROUGE** - Recall-oriented understudy for GISTING evaluation
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence

### Semantic Metrics (Optional)
- **BERTScore** - Uses contextual embeddings for semantic similarity

### Diagnostic Metrics
- **Length Ratio** - Checks if generated captions are proportional to references

## Quick Start

### Option 1: Integrated Evaluation (Recommended)

Generates captions AND evaluates them automatically:

```bash
python medical_image_captioning_with_eval.py
```

This script:
1. Loads medical images from ROCOv2
2. Generates captions using Qwen2-VL
3. Computes all metrics against original captions
4. Saves results with detailed metrics

### Option 2: Evaluate Existing Results

If you already have generated captions in a CSV file:

```python
from medical_image_metrics import evaluate_medical_captions

# Requires CSV with columns: 'original_caption' and 'generated_caption'
detailed_df, aggregate_stats = evaluate_medical_captions(
    'results_medical/medical_captions_train.csv',
    output_dir='results_medical',
    include_bert=False  # Set to True for semantic similarity (slower)
)
```

### Option 3: Single Caption Evaluation

Evaluate individual caption pairs:

```python
from medical_image_metrics import CaptioningMetrics

metrics = CaptioningMetrics()
scores = metrics.compute_all_metrics(
    reference="Chest X-ray showing normal cardiac silhouette",
    hypothesis="Normal chest X-ray with typical heart size",
    include_bert=False
)

print(scores)
# Output: {'BLEU': 0.45, 'METEOR': 0.52, 'ROUGE-1': 0.60, 'ROUGE-2': 0.25, ...}
```

### Option 4: Batch Evaluation

Evaluate multiple captions programmatically:

```python
from medical_image_metrics import MetricsAggregator

aggregator = MetricsAggregator()

references = [
    "CT scan showing acute subdural hematoma",
    "MRI brain with multiple sclerosis lesions"
]

hypotheses = [
    "CT brain imaging reveals acute subdural bleed",
    "Brain MRI demonstrates demyelinating disease"
]

detailed_df, aggregate_stats = aggregator.evaluate_batch(
    references,
    hypotheses,
    include_bert=False
)

# Print summary
aggregator.print_summary(aggregate_stats)

# Save results
aggregator.save_results(
    detailed_df,
    aggregate_stats,
    output_dir='results_medical',
    filename_prefix='batch_eval'
)
```

## Metric Interpretation

### BLEU Score (0-1, higher is better)
- Measures word overlap between generated and reference captions
- **0.3-0.4**: Weak similarity
- **0.4-0.5**: Moderate similarity
- **0.5-0.6**: Good similarity
- **0.6+**: Strong similarity

**Limitations**: Doesn't account for synonyms or word order variations

### METEOR (0-1, higher is better)
- Considers synonyms, stemming, and word order
- Generally higher than BLEU for same captions
- Better correlation with human judgment
- **0.4-0.5**: Moderate quality
- **0.5-0.6**: Good quality
- **0.6+**: Excellent quality

### ROUGE Scores (0-1, higher is better)
- **ROUGE-1**: Unigram overlap (individual word matches)
- **ROUGE-2**: Bigram overlap (two-word phrase matches)
- **ROUGE-L**: Longest common subsequence (sequence matching)

ROUGE-1 > ROUGE-2 > ROUGE-L typically

### BERTScore (0-1, higher is better) - Optional
- Uses BERT embeddings for semantic similarity
- Captures meaning beyond exact word matches
- **Precision**: How much of generated text is "contained in" reference
- **Recall**: How much of reference is "captured by" generated text
- **F1**: Harmonic mean of precision and recall

**Note**: BERTScore is slower (~1-2 seconds per pair). Only use if semantic evaluation is critical.

### Length Ratio
- Ratio of generated caption length to reference length
- **0.8-1.2**: Reasonable caption length
- **< 0.5 or > 2.0**: Indicates generation issues
  - Too short (< 0.5): Model may be under-generating
  - Too long (> 2.0): Model may be over-generating

## Configuration

### In `medical_image_captioning_with_eval.py`:

```python
class MedicalConfig:
    NUM_IMAGES = 50              # Images to process
    COMPUTE_BERT_SCORE = False   # Set to True for semantic evaluation
    SPLIT = 'train'              # 'train', 'validation', or 'test'
```

### Performance Notes:
- Standard metrics (BLEU, METEOR, ROUGE): ~5-10ms per pair
- BERTScore: ~1-2 seconds per pair
- For 1000 images: ~5-15 seconds without BERTScore, ~30-60 minutes with BERTScore

## Output Files

When you run evaluation, the following files are created:

### 1. Detailed Metrics CSV
**File**: `medical_{split}_detailed.csv`

Contains per-image metrics:
```
image_index | BLEU | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | Length_Ratio | BERTScore-F1 (optional)
```

### 2. Aggregate Statistics JSON
**File**: `medical_{split}_aggregate.json`

Contains mean, std, min, max for each metric:
```json
{
  "BLEU": {
    "mean": 0.45,
    "std": 0.12,
    "min": 0.15,
    "max": 0.78
  },
  ...
}
```

### 3. Summary Report
**File**: `medical_{split}_summary.txt`

Human-readable summary of all metrics.

### 4. Results with Metrics CSV
**File**: `medical_captions_{split}_with_metrics.csv`

Original results merged with computed metrics:
```
image_id | original_caption | generated_caption | BLEU | METEOR | ... | timestamp
```

## Installation

Install evaluation dependencies:

```bash
pip install -r requirements_evaluation.txt
```

Or individually:

```bash
# Essential metrics
pip install nltk rouge-score

# Optional semantic similarity
pip install bert-score
```

## Troubleshooting

### NLTK Data Missing
If you get "LookupError: Resource punkt not found", the script will auto-download, but you can manually do:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### BERTScore GPU Memory
If BERTScore crashes with OOM:
- Set `include_bert=False` in evaluation calls
- Or reduce batch size in BERTScore

### Empty Metrics
If metrics are all NaN or empty:
- Check CSV has 'original_caption' and 'generated_caption' columns
- Verify captions are strings, not null
- Check for "Error:" prefixed captions

## Medical-Specific Considerations

For medical image captioning, consider:

1. **Domain Terminology**: Models may struggle with specialized medical terms
   - METEOR's synonym matching helps partially
   - BERTScore may miss specialized medical terms

2. **Factual Accuracy**: Standard metrics don't measure clinical accuracy
   - Consider manual review of generated vs. reference captions
   - Look for medical errors beyond lexical differences

3. **Report Style**: Medical reports have specific structures
   - Length ratio helps detect structural differences
   - Manual review recommended for quality assessment

4. **Benchmarking**: For medical applications, typical ranges are:
   - BLEU: 0.25-0.45
   - METEOR: 0.35-0.55
   - ROUGE-L: 0.40-0.60

## Advanced Usage

### Custom Metric Evaluation

Add custom metrics to `medical_image_metrics.py`:

```python
class CaptioningMetrics:
    def compute_custom_metric(self, reference: str, hypothesis: str) -> float:
        # Your custom metric implementation
        return score
```

Then update `compute_all_metrics()` to include your metric.

### Filtering and Analysis

After evaluation, filter results in pandas:

```python
import pandas as pd

results = pd.read_csv('medical_captions_train_with_metrics.csv')

# Find low-quality captions
low_quality = results[results['BLEU'] < 0.3]
print(f"Low BLEU captions: {len(low_quality)}")

# Find too-short captions
short_captions = results[results['Length_Ratio'] < 0.5]
print(f"Too-short captions: {len(short_captions)}")

# Analyze by length
results['ref_length'] = results['original_caption'].apply(lambda x: len(x.split()))
print(results[['BLEU', 'METEOR', 'ref_length']].corr())
```

## References

- **BLEU**: Papineni et al. "BLEU: a Method for Automatic Evaluation of Machine Translation" (2002)
- **METEOR**: Banerjee & Lavie "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments" (2005)
- **ROUGE**: Lin "ROUGE: A Package for Automatic Evaluation of Summaries" (2004)
- **BERTScore**: Zhang et al. "BERTScore: Evaluating Text Generation with BERT" (2020)

## Support

For issues or questions:
1. Check error messages in output
2. Verify CSV format and column names
3. Test with smaller dataset first (`NUM_IMAGES = 10`)
4. Review metric computation in `medical_image_metrics.py`
