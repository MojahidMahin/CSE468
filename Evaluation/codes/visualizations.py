"""
Visualization Module for Evaluation Results

Creates various visualizations for analyzing model performance:
- Histograms of metric distributions
- Box plots comparing models
- Scatter plots for correlation analysis
- Heatmaps for metric comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Create visualizations for evaluation results"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics_histogram(self, df: pd.DataFrame, metric: str, model_name: str = None):
        """
        Plot histogram of a specific metric

        Args:
            df: DataFrame with results
            metric: Metric column name
            model_name: Optional model name for title
        """
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in dataframe")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(df[metric], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        title = f'Distribution of {metric}'
        if model_name:
            title += f' - {model_name}'
        plt.title(title, fontsize=14, fontweight='bold')

        plt.grid(axis='y', alpha=0.3)

        # Add statistics
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        plt.legend()

        filename = f'histogram_{metric}_{model_name}.png' if model_name else f'histogram_{metric}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    def plot_metrics_boxplot(self, results_dict: Dict[str, pd.DataFrame], metrics: List[str]):
        """
        Plot box plots comparing metrics across models

        Args:
            results_dict: Dict mapping model names to DataFrames
            metrics: List of metrics to plot
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))

        if num_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Prepare data for boxplot
            data_to_plot = []
            labels = []

            for model_name, df in results_dict.items():
                if metric in df.columns:
                    data_to_plot.append(df[metric].values)
                    labels.append(model_name)

            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

                # Color boxes
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)

                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'boxplot_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: boxplot_metrics_comparison.png")

    def plot_model_comparison_heatmap(self, summary_df: pd.DataFrame):
        """
        Plot heatmap comparing all models across all metrics

        Args:
            summary_df: DataFrame with models as rows and metrics as columns
        """
        plt.figure(figsize=(12, 6))

        # Normalize each metric to 0-1 for better visualization
        normalized_df = summary_df.copy()
        for col in normalized_df.columns:
            if col != 'model':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)

        # Set model as index
        if 'model' in normalized_df.columns:
            normalized_df = normalized_df.set_index('model')

        sns.heatmap(
            normalized_df,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Normalized Score'},
            linewidths=0.5
        )

        plt.title('Model Performance Heatmap (Normalized Metrics)', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'heatmap_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: heatmap_model_comparison.png")

    def plot_metric_correlation(self, df: pd.DataFrame, metrics: List[str]):
        """
        Plot correlation heatmap between different metrics

        Args:
            df: DataFrame with metric columns
            metrics: List of metrics to include
        """
        # Filter to only include available metrics
        available_metrics = [m for m in metrics if m in df.columns]

        if len(available_metrics) < 2:
            print("Warning: Need at least 2 metrics for correlation plot")
            return

        correlation_matrix = df[available_metrics].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'}
        )

        plt.title('Metric Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'heatmap_metric_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: heatmap_metric_correlation.png")

    def plot_performance_summary(self, summary_df: pd.DataFrame):
        """
        Plot bar chart summarizing model performance across key metrics

        Args:
            summary_df: DataFrame with summary statistics per model
        """
        # Select key metrics for summary
        key_metrics = ['BLEU-4', 'ROUGE-L', 'METEOR', 'BERTScore-F1']
        available_metrics = [m for m in key_metrics if m in summary_df.columns]

        if not available_metrics:
            print("Warning: No key metrics available for summary plot")
            return

        models = summary_df['model'].values if 'model' in summary_df.columns else summary_df.index

        x = np.arange(len(models))
        width = 0.2
        multiplier = 0

        fig, ax = plt.subplots(figsize=(12, 6))

        for metric in available_metrics:
            offset = width * multiplier
            values = summary_df[metric].values
            ax.bar(x + offset, values, width, label=metric)
            multiplier += 1

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Summary (Key Metrics)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left', ncol=2)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'bar_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: bar_performance_summary.png")

    def create_all_visualizations(self, results_dict: Dict[str, pd.DataFrame], summary_df: pd.DataFrame):
        """
        Create all visualizations at once

        Args:
            results_dict: Dict mapping model names to detailed results DataFrames
            summary_df: DataFrame with summary statistics
        """
        print("\nCreating visualizations...")

        # Key metrics to visualize
        key_metrics = ['BLEU-4', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'BERTScore-F1']

        # 1. Histograms for each model and metric
        print("\n1. Creating histograms...")
        for model_name, df in results_dict.items():
            for metric in key_metrics:
                if metric in df.columns:
                    self.plot_metrics_histogram(df, metric, model_name)

        # 2. Box plots comparing models
        print("\n2. Creating box plots...")
        self.plot_metrics_boxplot(results_dict, key_metrics)

        # 3. Heatmap of model comparison
        print("\n3. Creating model comparison heatmap...")
        self.plot_model_comparison_heatmap(summary_df)

        # 4. Metric correlation (using first model's data)
        print("\n4. Creating metric correlation heatmap...")
        first_model_df = list(results_dict.values())[0]
        all_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                       'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
                       'METEOR', 'ChrF', 'BERTScore-F1']
        self.plot_metric_correlation(first_model_df, all_metrics)

        # 5. Performance summary
        print("\n5. Creating performance summary...")
        self.plot_performance_summary(summary_df)

        print("\n✓ All visualizations created!\n")


if __name__ == "__main__":
    # Test visualization module with dummy data
    print("Testing Visualization Module...\n")

    try:
        from config import EvaluationConfig

        # Create dummy results
        np.random.seed(42)
        n_samples = 100

        models = ['Qwen2-VL-2B', 'Phi3-Vision', 'InternVL2-2B', 'SmolVLM2']
        results_dict = {}

        for i, model in enumerate(models):
            # Generate random scores (with slight variation per model)
            base_offset = i * 0.05
            results_dict[model] = pd.DataFrame({
                'image_id': [f'PMC_{j:05d}' for j in range(n_samples)],
                'BLEU-4': np.random.beta(5, 2, n_samples) * 0.6 + base_offset,
                'ROUGE-1': np.random.beta(6, 2, n_samples) * 0.7 + base_offset,
                'ROUGE-L': np.random.beta(5, 2, n_samples) * 0.6 + base_offset,
                'METEOR': np.random.beta(5, 3, n_samples) * 0.5 + base_offset,
                'BERTScore-F1': np.random.beta(7, 2, n_samples) * 0.8 + base_offset
            })

        # Create summary
        summary_data = []
        for model, df in results_dict.items():
            summary_data.append({
                'model': model,
                'BLEU-4': df['BLEU-4'].mean(),
                'ROUGE-1': df['ROUGE-1'].mean(),
                'ROUGE-L': df['ROUGE-L'].mean(),
                'METEOR': df['METEOR'].mean(),
                'BERTScore-F1': df['BERTScore-F1'].mean()
            })
        summary_df = pd.DataFrame(summary_data)

        # Create visualizations
        visualizer = ResultsVisualizer(EvaluationConfig.VISUALIZATIONS_DIR)
        visualizer.create_all_visualizations(results_dict, summary_df)

        print("✓ Visualization module test passed!")
        print(f"✓ Visualizations saved to: {EvaluationConfig.VISUALIZATIONS_DIR}")

    except Exception as e:
        print(f"\n✗ Visualization module test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
