# Ultra-Concise Prompts for Stubborn Models

## Problem
Current prompt asks for "1-2 sentences" but Qwen2-VL-2B still generates 3-4 sentences with hallucinations.

## Solutions

### Try These Prompts in Order:

#### 1. Strict One-Sentence Constraint
```python
ZERO_SHOT_PROMPT = (
    "Describe the key clinical finding in this medical image in ONE sentence only. "
    "Be specific and concise."
)
```

#### 2. Direct Command Format
```python
ZERO_SHOT_PROMPT = (
    "Write a one-sentence radiology finding for this image. "
    "Format: [Imaging modality] shows [specific clinical observation]."
)
```

#### 3. Example-Based Prompting
```python
ZERO_SHOT_PROMPT = (
    "Provide a one-sentence radiology finding. "
    "Example format: 'CT scan shows bilateral infiltrations and pleural effusion.' "
    "Your finding:"
)
```

#### 4. Minimal Prompt
```python
ZERO_SHOT_PROMPT = "One-sentence clinical finding:"
```

#### 5. Role + Template
```python
ZERO_SHOT_PROMPT = (
    "You are a radiologist. Write ONE sentence describing this image's key finding. "
    "Use format: [Modality] [shows/demonstrates/reveals] [finding]."
)
```

## Testing
```bash
# Edit config.py with each prompt and test:
python run_evaluation.py --num-images 5 --models Qwen2-VL-2B
```

## Expected Reality
With small general VLMs in zero-shot:
- BLEU-4: 0.05-0.15 (realistic max without fine-tuning)
- ROUGE-L: 0.15-0.30 (realistic max)
- BERTScore: 0.60-0.75 (realistic max)

You may not reach medical benchmark scores (0.10-0.30 BLEU) without:
- Medical-specific VLM (like LLaVA-Med, BiomedCLIP)
- Fine-tuning on radiology data
- Larger models (7B+)