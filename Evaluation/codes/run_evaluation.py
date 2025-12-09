"""
Main Evaluation Script

Runs zero-shot evaluation of VLM models on radiology dataset.

Usage:
    python run_evaluation.py [--num-images N] [--models MODEL1,MODEL2,...]

Example:
    python run_evaluation.py --num-images 500
    python run_evaluation.py --num-images 100 --models Qwen2-VL-2B,Phi3-Vision
"""

import argparse
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
import gc

from config import EvaluationConfig
from data_loader import RadiologydataLoader
from vlm_models import VLMModelFactory
from metrics import MetricsCalculator
from visualizations import ResultsVisualizer


class VLMEvaluationPipeline:
    """Main pipeline for VLM evaluation"""

    def __init__(self, config: EvaluationConfig = EvaluationConfig):
        self.config = config
        self.data_loader = None
        self.metrics_calculator = None
        self.visualizer = None

        # Results storage
        self.all_results = {}
        self.summary_results = []

    def initialize(self):
        """Initialize pipeline components"""
        print("\n" + "="*60)
        print("INITIALIZING VLM EVALUATION PIPELINE")
        print("="*60)

        # Validate config
        self.config.validate()
        self.config.print_config()

        # Initialize components
        print("Initializing components...")
        self.data_loader = RadiologydataLoader(self.config)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = ResultsVisualizer(self.config.VISUALIZATIONS_DIR)

        print("✓ Pipeline initialized successfully\n")

    def run_inference_single_model(self, model_config: dict) -> pd.DataFrame:
        """
        Run inference for a single model on all images

        Args:
            model_config: Model configuration dict

        Returns:
            DataFrame with results (image_id, image_path, generated_caption, ground_truth)
        """
        model_name = model_config['name']
        model_id = model_config['model_id']

        print("\n" + "="*60)
        print(f"RUNNING INFERENCE: {model_name}")
        print("="*60)

        # Load model
        model = VLMModelFactory.create_model(
            model_name=model_name,
            model_id=model_id,
            device=self.config.DEVICE,
            use_fp16=self.config.USE_FP16
        )
        model.load_model()

        # Get samples
        samples = self.data_loader.get_samples_by_limit(self.config.NUM_IMAGES)
        print(f"\nProcessing {len(samples)} images...")

        # Storage for results
        results = []

        # Process each image
        for idx, sample in enumerate(tqdm(samples, desc=f"Generating captions")):
            try:
                # Generate caption
                start_time = time.time()
                generated_caption = model.generate_caption(
                    sample['image'],
                    self.config.ZERO_SHOT_PROMPT
                )
                processing_time = time.time() - start_time

                # Store result
                results.append({
                    'image_id': sample['id'],
                    'image_path': sample['image_path'],
                    'generated_caption': generated_caption,
                    'ground_truth': sample['caption'],
                    'processing_time_sec': processing_time,
                    'timestamp': datetime.now().isoformat()
                })

                # Checkpoint every N images
                if (idx + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_path = self.config.OUTPUTS_DIR / f"checkpoint_{model_name}_{idx+1}.csv"
                    checkpoint_df.to_csv(checkpoint_path, index=False)
                    print(f"\n  ✓ Checkpoint saved: {checkpoint_path.name}")

            except Exception as e:
                print(f"\n  ✗ Error processing {sample['id']}: {e}")
                continue

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Save full results
        output_path = self.config.OUTPUTS_DIR / f"outputs_{model_name}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved: {output_path}")

        # Unload model
        model.unload_model()
        gc.collect()
        torch.cuda.empty_cache()

        return results_df

    def calculate_metrics_for_model(self, results_df: pd.DataFrame, model_name: str) -> dict:
        """
        Calculate all metrics for a model's results

        Args:
            results_df: DataFrame with generated_caption and ground_truth columns
            model_name: Name of the model

        Returns:
            Dict with metric scores
        """
        print(f"\n{'='*60}")
        print(f"CALCULATING METRICS: {model_name}")
        print('='*60)

        predictions = results_df['generated_caption'].tolist()
        references = results_df['ground_truth'].tolist()

        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(predictions, references)

        # Add processing time statistics
        metrics['avg_processing_time'] = results_df['processing_time_sec'].mean()
        metrics['total_processing_time'] = results_df['processing_time_sec'].sum()

        # Print metrics
        print("\nMetrics Summary:")
        print("-" * 60)
        for metric, value in metrics.items():
            print(f"  {metric:30s}: {value:.4f}")
        print("-" * 60)

        return metrics

    def add_metrics_to_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add per-sample metrics to results DataFrame

        Args:
            results_df: DataFrame with results

        Returns:
            DataFrame with added metric columns
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        from rouge_score import rouge_scorer

        print("\nCalculating per-sample metrics...")

        smoothing = SmoothingFunction().method1
        rouge_calc = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        bleu_scores = []
        rouge1_scores = []
        rougeL_scores = []

        for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Per-sample metrics"):
            pred = row['generated_caption']
            ref = row['ground_truth']

            # BLEU-4
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]
            bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)

            # ROUGE
            rouge_scores = rouge_calc.score(ref, pred)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        # Add to DataFrame
        results_df['BLEU-4'] = bleu_scores
        results_df['ROUGE-1'] = rouge1_scores
        results_df['ROUGE-L'] = rougeL_scores

        return results_df

    def run_full_evaluation(self):
        """Run complete evaluation pipeline for all models"""
        print("\n" + "="*60)
        print("STARTING FULL EVALUATION PIPELINE")
        print("="*60)

        enabled_models = self.config.get_enabled_models()

        for model_config in enabled_models:
            model_name = model_config['name']

            # Run inference
            results_df = self.run_inference_single_model(model_config)

            # Calculate aggregate metrics
            metrics = self.calculate_metrics_for_model(results_df, model_name)

            # Add per-sample metrics
            results_df = self.add_metrics_to_results(results_df)

            # Save detailed results with metrics
            detailed_output_path = self.config.OUTPUTS_DIR / f"detailed_results_{model_name}.csv"
            results_df.to_csv(detailed_output_path, index=False)
            print(f"✓ Detailed results saved: {detailed_output_path}")

            # Store for comparison
            self.all_results[model_name] = results_df
            metrics['model'] = model_name
            self.summary_results.append(metrics)

        # Create summary DataFrame
        summary_df = pd.DataFrame(self.summary_results)

        # Reorder columns
        metric_cols = [c for c in summary_df.columns if c != 'model']
        summary_df = summary_df[['model'] + metric_cols]

        # Save summary
        summary_path = self.config.METRICS_DIR / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved: {summary_path}")

        # Print final summary
        self.print_final_summary(summary_df)

        # Create visualizations
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        self.visualizer.create_all_visualizations(self.all_results, summary_df)

        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"\nResults saved in: {self.config.OUTPUT_DIR}")
        print(f"  - Model outputs: {self.config.OUTPUTS_DIR}")
        print(f"  - Metrics: {self.config.METRICS_DIR}")
        print(f"  - Visualizations: {self.config.VISUALIZATIONS_DIR}")

    def print_final_summary(self, summary_df: pd.DataFrame):
        """Print final summary table"""
        print("\n" + "="*60)
        print("FINAL EVALUATION SUMMARY")
        print("="*60 + "\n")

        # Key metrics to display
        key_metrics = ['model', 'BLEU-4', 'ROUGE-1', 'ROUGE-L', 'METEOR',
                       'BERTScore-F1', 'avg_processing_time']

        display_metrics = [m for m in key_metrics if m in summary_df.columns]
        display_df = summary_df[display_metrics]

        print(display_df.to_string(index=False))
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run VLM evaluation on radiology dataset")
    parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Number of images to process (default: from config)'
    )
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help='Comma-separated list of models to evaluate (default: all enabled)'
    )

    args = parser.parse_args()

    # Override config if specified
    if args.num_images:
        EvaluationConfig.NUM_IMAGES = args.num_images

    if args.models:
        selected_models = [m.strip() for m in args.models.split(',')]
        for model_config in EvaluationConfig.MODELS_CONFIG:
            model_config['enabled'] = model_config['name'] in selected_models

    # Run evaluation
    pipeline = VLMEvaluationPipeline(EvaluationConfig)
    pipeline.initialize()
    pipeline.run_full_evaluation()


if __name__ == "__main__":
    main()
