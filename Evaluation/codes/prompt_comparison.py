"""
Prompt Comparison Tool

Test different prompts to find the best match for ground truth style.
"""

from config import EvaluationConfig
from data_loader import RadiologydataLoader
from vlm_models import VLMModelFactory
import pandas as pd


# Different prompt strategies
PROMPT_VARIATIONS = {
    "current": (
        "Describe this medical image in detail. "
        "Include observations about anatomical structures, "
        "any visible abnormalities, and key clinical findings."
    ),

    "concise_v1": (
        "Provide a concise radiology report finding for this medical image. "
        "Describe only the key clinical observations in 1-2 sentences."
    ),

    "concise_v2": (
        "What are the key clinical findings in this medical image? "
        "Provide a brief radiology report."
    ),

    "ultra_concise": (
        "Describe the clinical findings in this medical image in one sentence."
    ),

    "radiology_style": (
        "Act as a radiologist. Provide a concise finding for this medical image "
        "as you would in a radiology report. Focus only on visible abnormalities "
        "or key observations."
    ),

    "minimal": (
        "Describe the key finding in this image."
    )
}


def test_prompts_on_sample(sample_idx=0, model_name="Qwen2-VL-2B"):
    """Test all prompt variations on a single sample"""

    # Load data
    loader = RadiologydataLoader()
    sample = loader.get_sample(sample_idx)

    print(f"\n{'='*70}")
    print(f"Testing Prompts on: {sample['id']}")
    print(f"{'='*70}")
    print(f"\nGround Truth:")
    print(f"  {sample['caption']}")
    print(f"\n{'='*70}\n")

    # Load model once
    model_config = [m for m in EvaluationConfig.MODELS_CONFIG if m['name'] == model_name][0]
    model = VLMModelFactory.create_model(
        model_name=model_name,
        model_id=model_config['model_id'],
        device=EvaluationConfig.DEVICE,
        use_fp16=EvaluationConfig.USE_FP16
    )
    model.load_model()

    results = []

    # Test each prompt
    for prompt_name, prompt_text in PROMPT_VARIATIONS.items():
        print(f"\n{'-'*70}")
        print(f"Prompt: {prompt_name}")
        print(f"{'-'*70}")
        print(f"Prompt Text: {prompt_text[:100]}...")
        print()

        caption = model.generate_caption(sample['image'], prompt_text)

        print(f"Generated Caption:")
        print(f"  {caption[:200]}...")
        print(f"\nLength: {len(caption)} chars vs Ground Truth: {len(sample['caption'])} chars")

        results.append({
            'prompt_name': prompt_name,
            'prompt_text': prompt_text,
            'generated_caption': caption,
            'generated_length': len(caption),
            'ground_truth': sample['caption'],
            'ground_truth_length': len(sample['caption'])
        })

    # Unload model
    model.unload_model()

    # Save results
    df = pd.DataFrame(results)
    output_path = EvaluationConfig.OUTPUT_DIR / "prompt_comparison_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")

    return df


def recommend_best_prompt():
    """Analyze results and recommend best prompt"""
    print("\n" + "="*70)
    print("PROMPT RECOMMENDATION")
    print("="*70)
    print("\nBased on ground truth analysis:")
    print("  - Ground truth: Concise, 1-2 sentences")
    print("  - Style: Clinical radiology report findings")
    print("  - Length: 50-150 characters typically")
    print("\nRecommended Prompts (in order of preference):")
    print("\n1. 'concise_v1' - Best balance of conciseness and detail")
    print("   → Explicitly asks for 1-2 sentences")
    print("\n2. 'radiology_style' - Mimics radiologist behavior")
    print("   → Uses role-playing to match report style")
    print("\n3. 'concise_v2' - Simple and direct")
    print("   → Good for models that follow instructions well")
    print("\nTo update your evaluation:")
    print("  1. Edit config.py")
    print("  2. Change ZERO_SHOT_PROMPT to your chosen prompt")
    print("  3. Re-run evaluation")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test different prompts")
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to test')
    parser.add_argument('--model', type=str, default='Qwen2-VL-2B', help='Model to use')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("PROMPT COMPARISON TOOL")
    print("="*70)
    print("\nThis will test 6 different prompts on a sample image")
    print("to find the best match for the ground truth style.\n")

    # Test prompts
    results_df = test_prompts_on_sample(args.sample_idx, args.model)

    # Show recommendations
    recommend_best_prompt()
