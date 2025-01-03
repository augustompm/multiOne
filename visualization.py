"""
Visualization script for Multiple Sequence Alignment results
Creates various plots to analyze the performance of ClustalW and MUSCLE
compared to BAliBASE reference alignments
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import List, Dict
import logging

class AlignmentVisualizer:
    def __init__(self, results_dir: str = "/dados/home/tesla-dados/multione/results"):
        """
        Initialize visualizer with paths
        
        Args:
            results_dir: Directory containing results data
        """
        self.results_dir = Path(results_dir)
        self.data_dir = self.results_dir / "data"
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for all plots
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def load_results(self) -> pd.DataFrame:
        """
        Load results from CSV file
        
        Returns:
            DataFrame containing alignment results
        """
        try:
            results_file = self.data_dir / "alignment_results.csv"
            df = pd.DataFrame(pd.read_csv(results_file))
            return df
        except Exception as e:
            logging.error(f"Failed to load results: {str(e)}")
            return pd.DataFrame()

    def create_score_distribution_plot(self, df: pd.DataFrame):
        """
        Create violin plots showing score distributions for each method
        
        Args:
            df: DataFrame containing results
        """
        metrics = ['SP', 'TC', 'SPW', 'Rodrigo']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Score Distributions by Alignment Method', fontsize=16, y=1.02)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            sns.violinplot(data=df, x='Method', y=metric, ax=ax)
            ax.set_title(f'{metric} Score Distribution')
            ax.set_xlabel('Alignment Method')
            ax.set_ylabel(f'{metric} Score')
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_heatmap(self, df: pd.DataFrame):
        """
        Create heatmap comparing methods across metrics
        
        Args:
            df: DataFrame containing results
        """
        # Calculate mean scores for each method and metric
        metrics = ['SP', 'TC', 'SPW', 'Rodrigo']
        methods = df['Method'].unique()
        
        means = []
        for method in methods:
            method_means = [df[df['Method'] == method][metric].mean() for metric in metrics]
            means.append(method_means)
            
        mean_matrix = np.array(means)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_matrix, 
                   annot=True, 
                   fmt='.3f',
                   xticklabels=metrics,
                   yticklabels=methods,
                   cmap='YlOrRd')
        
        plt.title('Average Performance by Method and Metric')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_sequence_performance_plot(self, df: pd.DataFrame):
        """
        Create line plot showing performance across different sequences
        
        Args:
            df: DataFrame containing results
        """
        metrics = ['SP', 'TC', 'SPW', 'Rodrigo']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Across Sequences', fontsize=16, y=1.02)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            for method in df['Method'].unique():
                method_data = df[df['Method'] == method]
                ax.plot(method_data['Sequence'], 
                       method_data[metric], 
                       marker='o', 
                       label=method)
                
            ax.set_title(f'{metric} Scores by Sequence')
            ax.set_xlabel('Sequence')
            ax.set_ylabel(f'{metric} Score')
            ax.tick_params(axis='x', rotation=90)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sequence_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_method_comparison_plot(self, df: pd.DataFrame):
        """
        Create scatter plots comparing methods to BAliBASE reference
        
        Args:
            df: DataFrame containing results
        """
        metrics = ['SP', 'TC', 'SPW', 'Rodrigo']
        methods = ['ClustalW', 'MUSCLE']
        
        fig, axes = plt.subplots(len(methods), len(metrics), figsize=(20, 10))
        fig.suptitle('Method Comparison with BAliBASE Reference', fontsize=16, y=1.02)
        
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                
                method_scores = df[df['Method'] == method][metric]
                balibase_scores = df[df['Method'] == 'BAliBASE'][metric]
                
                ax.scatter(balibase_scores, method_scores, alpha=0.6)
                
                # Add diagonal line for reference
                min_val = min(method_scores.min(), balibase_scores.min())
                max_val = max(method_scores.max(), balibase_scores.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                ax.set_title(f'{method} vs BAliBASE: {metric}')
                ax.set_xlabel('BAliBASE Score')
                ax.set_ylabel(f'{method} Score')
                
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_statistical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical summary of results
        
        Args:
            df: DataFrame containing results
            
        Returns:
            DataFrame containing statistical summary
        """
        metrics = ['SP', 'TC', 'SPW', 'Rodrigo']
        methods = df['Method'].unique()
        
        stats_data = []
        for method in methods:
            method_data = df[df['Method'] == method]
            
            for metric in metrics:
                stat_row = {
                    'Method': method,
                    'Metric': metric,
                    'Mean': method_data[metric].mean(),
                    'Std': method_data[metric].std(),
                    'Min': method_data[metric].min(),
                    'Max': method_data[metric].max(),
                    'Median': method_data[metric].median()
                }
                stats_data.append(stat_row)
                
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(self.plots_dir / 'statistical_summary.csv', index=False)
        return stats_df

    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        try:
            # Load results
            df = self.load_results()
            if df.empty:
                logging.error("No results data found")
                return
                
            # Create all plots
            logging.info("Generating visualizations...")
            self.create_score_distribution_plot(df)
            self.create_performance_heatmap(df)
            self.create_sequence_performance_plot(df)
            self.create_method_comparison_plot(df)
            
            # Generate statistical summary
            stats_df = self.create_statistical_summary(df)
            logging.info("Statistical summary saved to statistical_summary.csv")
            
            logging.info(f"All visualizations saved to {self.plots_dir}")
            
        except Exception as e:
            logging.error(f"Failed to generate visualizations: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    visualizer = AlignmentVisualizer()
    visualizer.generate_all_visualizations()