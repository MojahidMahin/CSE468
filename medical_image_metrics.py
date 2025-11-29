"""
Evaluation metrics for medical image captioning.
Provides BLEU, METEOR, CIDEr, ROUGE, and semantic similarity metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class CaptioningMetrics:
    """Compute standard evaluation metrics for image captioning."""

    def __init__(self):
        """Initialize metrics object with required libraries."""
        self._init_metrics()

    def _init_metrics(self):
        """Lazy-load metric libraries to avoid unnecessary imports."""
        self.bleu = None
        self.meteor = None
        self.cider = None
        self.rouge = None
        self.bert_score = None

    def _load_bleu(self):
        """Load BLEU metric from nltk."""
        if self.bleu is None:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            self.bleu = {'func': sentence_bleu, 'tokenize': word_tokenize}
        return self.bleu

    def _load_meteor(self):
        """Load METEOR metric from nltk."""
        if self.meteor is None:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
            self.meteor = meteor_score
        return self.meteor

    def _load_rouge(self):
        """Load ROUGE metric."""
        if self.rouge is None:
            try:
                from rouge_score import rouge_scorer
                self.rouge = rouge_scorer
            except ImportError:
                print("Warning: rouge-score not installed. ROUGE metrics will be skipped.")
                self.rouge = False
        return self.rouge

    def _load_bert_score(self):
        """Load BERTScore for semantic similarity."""
        if self.bert_score is None:
            try:
                import bert_score
                self.bert_score = bert_score
            except ImportError:
                print("Warning: bert-score not installed. Semantic similarity will be skipped.")
                self.bert_score = False
        return self.bert_score

    def compute_bleu(self, reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
        """
        Compute BLEU score (higher is better, 0-1 range).
        Measures n-gram overlap with reference captions.

        Args:
            reference: Reference caption text
            hypothesis: Generated caption text
            max_n: Maximum n-gram order (default: 4 for BLEU-4)

        Returns:
            Dictionary with BLEU scores for each n-gram order
        """
        try:
            bleu_module = self._load_bleu()
            if not bleu_module:
                return {}

            tokenize = bleu_module['tokenize']
            sentence_bleu = bleu_module['func']

            ref_tokens = tokenize(reference.lower())
            hyp_tokens = tokenize(hypothesis.lower())

            weights = tuple([1.0 / max_n] * max_n)
            smoothing_function = __import__('nltk.translate.bleu_score', fromlist=['SmoothingFunction']).SmoothingFunction().method1

            bleu = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=smoothing_function
            )

            return {'BLEU': bleu}
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            return {}

    def compute_meteor(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute METEOR score (higher is better, 0-1 range).
        Considers synonyms and word stems in addition to exact matches.

        Args:
            reference: Reference caption text
            hypothesis: Generated caption text

        Returns:
            Dictionary with METEOR score
        """
        try:
            meteor_func = self._load_meteor()
            if not meteor_func:
                return {}

            score = meteor_func(reference.lower(), hypothesis.lower())
            return {'METEOR': score}
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            return {}

    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute ROUGE scores (higher is better, 0-1 range).
        ROUGE-1: Unigram overlap
        ROUGE-2: Bigram overlap
        ROUGE-L: Longest common subsequence

        Args:
            reference: Reference caption text
            hypothesis: Generated caption text

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        try:
            rouge_module = self._load_rouge()
            if not rouge_module:
                return {}

            scorer = rouge_module.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference.lower(), hypothesis.lower())

            return {
                'ROUGE-1': scores['rouge1'].fmeasure,
                'ROUGE-2': scores['rouge2'].fmeasure,
                'ROUGE-L': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return {}

    def compute_bert_score(self, reference: str, hypothesis: str, model_type: str = 'distilbert-base-uncased') -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity (higher is better, 0-1 range).
        Uses contextual embeddings to measure semantic similarity.

        Args:
            reference: Reference caption text
            hypothesis: Generated caption text
            model_type: BERTScore model to use

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        try:
            bert_module = self._load_bert_score()
            if not bert_module:
                return {}

            P, R, F1 = bert_module.score(
                [hypothesis],
                [reference],
                model_type=model_type,
                device='cuda',
                verbose=False
            )

            return {
                'BERTScore-Precision': P.item(),
                'BERTScore-Recall': R.item(),
                'BERTScore-F1': F1.item()
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {}

    def compute_length_ratio(self, reference: str, hypothesis: str) -> float:
        """
        Compute ratio of generated caption length to reference length.
        Useful for detecting if model generates too short/long captions.

        Args:
            reference: Reference caption text
            hypothesis: Generated caption text

        Returns:
            Ratio of hypothesis length to reference length
        """
        ref_len = len(reference.split())
        hyp_len = len(hypothesis.split())
        return hyp_len / ref_len if ref_len > 0 else 0.0

    def compute_all_metrics(self, reference: str, hypothesis: str, include_bert: bool = False) -> Dict[str, float]:
        """
        Compute all available metrics for a caption pair.

        Args:
            reference: Reference caption text
            hypothesis: Generated caption text
            include_bert: Whether to compute BERTScore (slower)

        Returns:
            Dictionary with all computed metrics
        """
        all_metrics = {}

        # Basic metrics
        all_metrics.update(self.compute_bleu(reference, hypothesis))
        all_metrics.update(self.compute_meteor(reference, hypothesis))
        all_metrics.update(self.compute_rouge(reference, hypothesis))

        # Length analysis
        all_metrics['Length_Ratio'] = self.compute_length_ratio(reference, hypothesis)

        # Semantic similarity (optional, slower)
        if include_bert:
            all_metrics.update(self.compute_bert_score(reference, hypothesis))

        return all_metrics


class MetricsAggregator:
    """Aggregate metrics across multiple captions."""

    def __init__(self):
        """Initialize aggregator."""
        self.metrics_computer = CaptioningMetrics()

    def evaluate_batch(self,
                      references: List[str],
                      hypotheses: List[str],
                      include_bert: bool = False) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Evaluate multiple caption pairs and compute aggregate statistics.

        Args:
            references: List of reference captions
            hypotheses: List of generated captions
            include_bert: Whether to compute BERTScore

        Returns:
            Tuple of (detailed metrics dataframe, aggregate statistics)
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")

        detailed_metrics = []

        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            metrics = self.metrics_computer.compute_all_metrics(ref, hyp, include_bert=include_bert)
            metrics['image_index'] = i
            detailed_metrics.append(metrics)

        metrics_df = pd.DataFrame(detailed_metrics)

        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_stats(metrics_df)

        return metrics_df, aggregate_stats

    def _compute_aggregate_stats(self, metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute mean, std, min, max for all metrics.

        Args:
            metrics_df: DataFrame with per-image metrics

        Returns:
            Dictionary with aggregate statistics
        """
        aggregate = {}
        metric_cols = [col for col in metrics_df.columns if col != 'image_index']

        for metric_col in metric_cols:
            if metric_col in metrics_df.columns:
                values = metrics_df[metric_col].dropna()
                if len(values) > 0:
                    aggregate[metric_col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }

        return aggregate

    def print_summary(self, aggregate_stats: Dict[str, Dict[str, float]], title: str = "Evaluation Results"):
        """
        Print formatted summary of evaluation metrics.

        Args:
            aggregate_stats: Aggregate statistics dictionary
            title: Title for the report
        """
        print("\n" + "=" * 100)
        print(f"{title}")
        print("=" * 100)

        for metric, stats in sorted(aggregate_stats.items()):
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        print("\n" + "=" * 100)

    def save_results(self,
                    detailed_df: pd.DataFrame,
                    aggregate_stats: Dict[str, Dict[str, float]],
                    output_dir: str,
                    filename_prefix: str = 'evaluation'):
        """
        Save evaluation results to files.

        Args:
            detailed_df: Detailed metrics DataFrame
            aggregate_stats: Aggregate statistics
            output_dir: Output directory path
            filename_prefix: Prefix for output files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save detailed metrics
        detailed_file = Path(output_dir) / f'{filename_prefix}_detailed.csv'
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Saved detailed metrics to: {detailed_file}")

        # Save aggregate statistics
        aggregate_file = Path(output_dir) / f'{filename_prefix}_aggregate.json'
        with open(aggregate_file, 'w') as f:
            json.dump(aggregate_stats, f, indent=2)
        print(f"Saved aggregate statistics to: {aggregate_file}")

        # Save summary report
        summary_file = Path(output_dir) / f'{filename_prefix}_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Report - {datetime.utcnow().isoformat()}\n")
            f.write("=" * 100 + "\n\n")

            for metric, stats in sorted(aggregate_stats.items()):
                f.write(f"{metric}:\n")
                f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")

        print(f"Saved summary report to: {summary_file}")


def evaluate_medical_captions(results_csv: str,
                             output_dir: str = None,
                             include_bert: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate medical image captions from results CSV.

    Args:
        results_csv: Path to CSV file with 'original_caption' and 'generated_caption' columns
        output_dir: Output directory for metrics (default: same as results CSV)
        include_bert: Whether to compute BERTScore

    Returns:
        Tuple of (detailed metrics DataFrame, aggregate statistics)
    """
    # Load results
    results_df = pd.read_csv(results_csv)

    if 'original_caption' not in results_df.columns or 'generated_caption' not in results_df.columns:
        raise ValueError("CSV must contain 'original_caption' and 'generated_caption' columns")

    # Compute metrics
    aggregator = MetricsAggregator()
    detailed_df, aggregate_stats = aggregator.evaluate_batch(
        results_df['original_caption'].tolist(),
        results_df['generated_caption'].tolist(),
        include_bert=include_bert
    )

    # Set output directory
    if output_dir is None:
        output_dir = str(Path(results_csv).parent)

    # Save results
    filename_prefix = Path(results_csv).stem
    aggregator.save_results(detailed_df, aggregate_stats, output_dir, filename_prefix)

    # Print summary
    aggregator.print_summary(aggregate_stats)

    return detailed_df, aggregate_stats


if __name__ == '__main__':
    # Example usage
    print("Medical Image Captioning Evaluation Metrics Module")
    print("=" * 100)
    print("\nUsage:")
    print("  1. From results CSV:")
    print("     from medical_image_metrics import evaluate_medical_captions")
    print("     detailed_df, aggregate_stats = evaluate_medical_captions('results.csv')")
    print("\n  2. For single caption pair:")
    print("     from medical_image_metrics import CaptioningMetrics")
    print("     metrics = CaptioningMetrics()")
    print("     scores = metrics.compute_all_metrics(reference, hypothesis)")
    print("\n  3. For batch evaluation:")
    print("     from medical_image_metrics import MetricsAggregator")
    print("     aggregator = MetricsAggregator()")
    print("     detailed_df, stats = aggregator.evaluate_batch(references, hypotheses)")
