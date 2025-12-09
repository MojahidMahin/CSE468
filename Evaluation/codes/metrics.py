"""
Evaluation Metrics Module

Implements various metrics for evaluating generated captions:
- BLEU (for translation quality)
- ROUGE-1/2/L (for summarization quality)
- METEOR (for semantic similarity)
- ChrF (character n-gram F-score)
- BERTScore (contextual embeddings similarity)
- Perplexity (language model perplexity)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calculate various evaluation metrics for generated captions"""

    def __init__(self):
        self.metrics_cache = {}
        self._initialize_scorers()

    def _initialize_scorers(self):
        """Initialize metric calculation libraries"""
        try:
            # BLEU, METEOR
            import nltk

            # Download all required NLTK data
            required_nltk_data = [
                ('tokenizers/punkt_tab', 'punkt_tab'),
                ('tokenizers/punkt', 'punkt'),
                ('corpora/wordnet', 'wordnet'),
                ('corpora/omw-1.4', 'omw-1.4')
            ]

            for resource_path, resource_name in required_nltk_data:
                try:
                    nltk.data.find(resource_path)
                except LookupError:
                    print(f"Downloading NLTK {resource_name}...")
                    nltk.download(resource_name, quiet=True)

            # BERTScore
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(
                model_type="microsoft/deberta-xlarge-mnli",
                lang="en",
                rescale_with_baseline=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("✓ Metrics scorers initialized")

        except Exception as e:
            print(f"Warning: Some metrics may not be available: {e}")

    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BLEU scores (BLEU-1 to BLEU-4)

        Args:
            predictions: List of generated captions
            references: List of ground truth captions

        Returns:
            Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize

        smoothing = SmoothingFunction().method1
        bleu_scores = {f"BLEU-{i}": [] for i in range(1, 5)}

        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]

            for n in range(1, 5):
                weights = tuple([1.0 / n] * n + [0] * (4 - n))
                score = sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    weights=weights,
                    smoothing_function=smoothing
                )
                bleu_scores[f"BLEU-{n}"].append(score)

        # Average scores
        return {k: np.mean(v) for k, v in bleu_scores.items()}

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)

        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        rouge_scores = defaultdict(list)

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge_scores['ROUGE-1'].append(scores['rouge1'].fmeasure)
            rouge_scores['ROUGE-2'].append(scores['rouge2'].fmeasure)
            rouge_scores['ROUGE-L'].append(scores['rougeL'].fmeasure)

        return {k: np.mean(v) for k, v in rouge_scores.items()}

    def calculate_meteor(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate METEOR score

        Returns:
            Average METEOR score
        """
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)

        return np.mean(scores)

    def calculate_chrf(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate ChrF (Character n-gram F-score)

        Returns:
            Average ChrF score
        """
        try:
            from sacrebleu.metrics import CHRF
            chrf = CHRF()

            # sacrebleu expects references as list of lists
            refs_formatted = [[ref] for ref in references]

            scores = []
            for pred, ref_list in zip(predictions, refs_formatted):
                score = chrf.sentence_score(pred, ref_list)
                scores.append(score.score / 100.0)  # Normalize to 0-1

            return np.mean(scores)

        except ImportError:
            print("Warning: sacrebleu not available, skipping ChrF")
            return 0.0

    def calculate_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore (Precision, Recall, F1)

        Returns:
            Dict with BERTScore-P, BERTScore-R, BERTScore-F1
        """
        try:
            P, R, F1 = self.bert_scorer.score(predictions, references)

            return {
                'BERTScore-P': P.mean().item(),
                'BERTScore-R': R.mean().item(),
                'BERTScore-F1': F1.mean().item()
            }

        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {e}")
            return {
                'BERTScore-P': 0.0,
                'BERTScore-R': 0.0,
                'BERTScore-F1': 0.0
            }

    def calculate_perplexity(self, texts: List[str]) -> float:
        """
        Calculate perplexity using a pre-trained language model

        Args:
            texts: List of generated captions

        Returns:
            Average perplexity score
        """
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            model_id = "gpt2"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
            model.eval()

            perplexities = []

            for text in texts:
                encodings = tokenizer(text, return_tensors="pt")
                max_length = min(encodings.input_ids.size(1), 1024)

                with torch.no_grad():
                    outputs = model(
                        encodings.input_ids[:, :max_length].to(device),
                        labels=encodings.input_ids[:, :max_length].to(device)
                    )
                    loss = outputs.loss

                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

            # Cleanup
            del model
            del tokenizer
            torch.cuda.empty_cache()

            return np.mean(perplexities)

        except Exception as e:
            print(f"Warning: Perplexity calculation failed: {e}")
            return 0.0

    def calculate_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate all metrics at once

        Args:
            predictions: List of generated captions
            references: List of ground truth captions

        Returns:
            Dict with all metric scores
        """
        print("\nCalculating metrics...")
        results = {}

        # BLEU
        print("  - BLEU...")
        bleu_scores = self.calculate_bleu(predictions, references)
        results.update(bleu_scores)

        # ROUGE
        print("  - ROUGE...")
        rouge_scores = self.calculate_rouge(predictions, references)
        results.update(rouge_scores)

        # METEOR
        print("  - METEOR...")
        results['METEOR'] = self.calculate_meteor(predictions, references)

        # ChrF
        print("  - ChrF...")
        results['ChrF'] = self.calculate_chrf(predictions, references)

        # BERTScore
        print("  - BERTScore...")
        bertscore = self.calculate_bertscore(predictions, references)
        results.update(bertscore)

        # Perplexity
        print("  - Perplexity...")
        results['Perplexity'] = self.calculate_perplexity(predictions)

        print("✓ All metrics calculated\n")
        return results


if __name__ == "__main__":
    # Test metrics module
    print("Testing Metrics Module...\n")

    try:
        # Sample predictions and references
        predictions = [
            "Chest X-ray shows bilateral infiltrates in the lungs",
            "CT scan reveals a mass in the right kidney",
            "Ultrasound shows normal blood flow patterns"
        ]

        references = [
            "Chest X-ray shows fine bilateral reticulo-interstitial infiltrates",
            "Enhanced CT showing a 2-cm mass lesion in right kidney",
            "Color Doppler ultrasound shows normal flow signal"
        ]

        # Initialize calculator
        calculator = MetricsCalculator()

        # Calculate all metrics
        results = calculator.calculate_all_metrics(predictions, references)

        # Print results
        print("Metric Results:")
        print("-" * 40)
        for metric, score in results.items():
            print(f"  {metric:20s}: {score:.4f}")

        print("\n✓ Metrics module test passed!")

    except Exception as e:
        print(f"\n✗ Metrics module test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
