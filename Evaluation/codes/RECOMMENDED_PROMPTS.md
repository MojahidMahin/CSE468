# Recommended Prompts for Radiology Evaluation

## Problem Analysis

**Ground Truth Style:**
- Concise (50-150 chars typically)
- 1-2 sentences
- Direct clinical findings
- Radiology report style
- Examples:
  - "Chest X-ray shows fine bilateral reticulo-interstitial infiltrates"
  - "CT angiography showing left upper lobe thick-walled cavity 4.8 x 3.4 cm"
  - "Thoracic CT scan showing bilateral pneumonic infiltrations and pleural effusion"

**Current Output Style:**
- Very long (500+ chars)
- Multiple paragraphs
- Educational explanations
- Discusses normal anatomy
- Low metrics: BLEU-4: 0.0021, ROUGE-1: 0.0578

## Recommended Prompts (Ranked)

### 🥇 Option 1: Concise Clinical (RECOMMENDED)
```python
ZERO_SHOT_PROMPT = (
    "Provide a concise radiology report finding for this medical image. "
    "Describe only the key clinical observations in 1-2 sentences."
)
```

**Why:** Explicitly requests 1-2 sentences and uses "radiology report" to set the tone.

---

### 🥈 Option 2: Radiologist Role-Play
```python
ZERO_SHOT_PROMPT = (
    "Act as a radiologist. Provide a concise finding for this medical image "
    "as you would in a radiology report. Focus only on visible abnormalities "
    "or key observations."
)
```

**Why:** Role-playing often helps models match professional writing styles.

---

### 🥉 Option 3: Ultra-Concise
```python
ZERO_SHOT_PROMPT = (
    "Describe the key clinical findings in this medical image in one sentence."
)
```

**Why:** Very direct, forces brevity. May sacrifice some detail.

---

### Option 4: Clinical Findings Focus
```python
ZERO_SHOT_PROMPT = (
    "What are the key clinical findings in this medical image? "
    "Provide a brief radiology report in 1-2 sentences."
)
```

**Why:** Question format can work well with instruction-tuned models.

---

### Option 5: Minimal
```python
ZERO_SHOT_PROMPT = "Describe the key finding in this image."
```

**Why:** Shortest prompt. Good for models that over-explain with longer prompts.

---

## How to Update

### Step 1: Edit config.py
```bash
nano /home/vortex/CSE\ 468\ AFE/Project/Evaluation/codes/config.py
```

### Step 2: Replace the ZERO_SHOT_PROMPT
Find line ~68 and replace with your chosen prompt.

### Step 3: Test on Small Sample
```bash
cd "/home/vortex/CSE 468 AFE/Project/Evaluation/codes"
python run_evaluation.py --num-images 10 --models Qwen2-VL-2B
```

### Step 4: Compare Metrics
Check if BLEU/ROUGE scores improve:
- Current: BLEU-4: 0.0021, ROUGE-1: 0.0578
- Expected improvement: 3-5x higher scores with concise prompt

### Step 5: Run Full Evaluation
Once satisfied with the prompt:
```bash
python run_evaluation.py --num-images 500
```

---

## Testing Different Prompts (After current run finishes)

Use the provided tool to test all prompts:
```bash
cd "/home/vortex/CSE 468 AFE/Project/Evaluation/codes"
python prompt_comparison.py --sample-idx 0
```

This will:
1. Test all 6 prompts on one image
2. Show generated captions side-by-side
3. Compare lengths with ground truth
4. Save results to CSV

---

## Expected Impact

With the recommended prompt, you should see:
- ✅ **Shorter captions** (100-200 chars vs 500+ chars)
- ✅ **Better BLEU scores** (0.01-0.05 vs 0.002)
- ✅ **Better ROUGE scores** (0.15-0.25 vs 0.06)
- ✅ **More relevant clinical observations**
- ✅ **Less verbose explanations**

---

## My Recommendation

**Use Option 1** for your 500-image evaluation:

```python
ZERO_SHOT_PROMPT = (
    "Provide a concise radiology report finding for this medical image. "
    "Describe only the key clinical observations in 1-2 sentences."
)
```

This balances:
- Conciseness (matches ground truth length)
- Clinical relevance (radiologist perspective)
- Specificity (1-2 sentences explicit)
- Clarity (direct instruction)
