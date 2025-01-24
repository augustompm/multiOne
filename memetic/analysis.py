# memetic/analysis.py

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

class ExecutionAnalyzer:
    """
    Analisa execuções do algoritmo fornecendo métricas e visualizações.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executions = []
        self.best_global_score = float('-inf')
        self.best_global_matrix = None
        self.convergence_data = defaultdict(list)
        
    def record_execution(
        self,
        initial_score: float,
        final_score: float,
        improvements: List[Dict],
        final_matrix: np.ndarray,
        execution_metadata: Optional[Dict] = None
    ) -> None:
        """
        Registra dados de uma execução para análise.
        
        Args:
            initial_score: Score inicial
            final_score: Score final
            improvements: Lista de melhorias encontradas
            final_matrix: Matriz final
            execution_metadata: Metadados adicionais da execução
        """
        execution = {
            'initial_score': initial_score,
            'final_score': final_score,
            'improvement_path': [imp['score_improvement'] for imp in improvements],
            'neighborhood_usage': self._analyze_neighborhoods(improvements),
            'convergence_profile': self._calculate_convergence(improvements),
            'final_matrix': final_matrix.copy(),
            'metadata': execution_metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.executions.append(execution)
        
        # Atualiza melhor global
        if final_score > self.best_global_score:
            self.best_global_score = final_score
            self.best_global_matrix = final_matrix.copy()
            self.logger.info(f"New best global score: {self.best_global_score:.4f}")
            
    def _analyze_neighborhoods(self, improvements: List[Dict]) -> Dict:
        """Analisa efetividade das vizinhanças."""
        neighborhood_stats = defaultdict(lambda: {'count': 0, 'total_improvement': 0.0})
        
        for imp in improvements:
            n_type = imp['neighborhood']
            neighborhood_stats[n_type]['count'] += 1
            neighborhood_stats[n_type]['total_improvement'] += imp['score_improvement']
            
        # Calcula efetividade
        for stats in neighborhood_stats.values():
            stats['avg_improvement'] = (
                stats['total_improvement'] / stats['count'] 
                if stats['count'] > 0 else 0.0
            )
            
        return dict(neighborhood_stats)
        
    def _calculate_convergence(self, improvements: List[Dict]) -> List[float]:
        """Calcula perfil de convergência."""
        cumulative_improvement = 0.0
        profile = []
        
        for imp in improvements:
            cumulative_improvement += imp['score_improvement']
            profile.append(cumulative_improvement)
            
        return profile
        
    def analyze_convergence(self) -> Dict:
        """Analisa padrões de convergência entre execuções."""
        if not self.executions:
            return {}
            
        convergence_stats = {
            'avg_initial_score': np.mean([ex['initial_score'] for ex in self.executions]),
            'avg_final_score': np.mean([ex['final_score'] for ex in self.executions]),
            'best_score': max(ex['final_score'] for ex in self.executions),
            'worst_score': min(ex['final_score'] for ex in self.executions),
            'avg_improvements': np.mean([
                len(ex['improvement_path']) for ex in self.executions
            ]),
            'early_convergence_rate': self._calculate_early_convergence_rate()
        }
        
        return convergence_stats
        
    def _calculate_early_convergence_rate(self, threshold: float = 0.9) -> float:
        """Calcula taxa de convergência precoce."""
        early_converged = 0
        
        for ex in self.executions:
            if ex['improvement_path']:
                total_improvement = sum(ex['improvement_path'])
                halfway_point = len(ex['improvement_path']) // 2
                early_improvement = sum(ex['improvement_path'][:halfway_point])
                
                if early_improvement > threshold * total_improvement:
                    early_converged += 1
                    
        return early_converged / len(self.executions) if self.executions else 0.0
        
    def analyze_neighborhood_effectiveness(self) -> Dict:
        """Analisa efetividade relativa das vizinhanças."""
        if not self.executions:
            return {}
            
        combined_stats = defaultdict(lambda: {'total_improvements': 0, 'total_value': 0.0})
        
        for ex in self.executions:
            for n_type, stats in ex['neighborhood_usage'].items():
                combined_stats[n_type]['total_improvements'] += stats['count']
                combined_stats[n_type]['total_value'] += stats['total_improvement']
                
        # Calcula efetividade relativa
        total_improvements = sum(
            stats['total_improvements'] for stats in combined_stats.values()
        )
        
        effectiveness = {}
        for n_type, stats in combined_stats.items():
            effectiveness[n_type] = {
                'usage_rate': stats['total_improvements'] / total_improvements,
                'avg_improvement': (
                    stats['total_value'] / stats['total_improvements']
                    if stats['total_improvements'] > 0 else 0.0
                )
            }
            
        return effectiveness
        
    def visualize_convergence(self, save_path: Path) -> None:
        """Gera visualização da convergência."""
        plt.figure(figsize=(12, 6))
        
        # Plot individual executions
        for i, ex in enumerate(self.executions):
            profile = ex['convergence_profile']
            iterations = range(1, len(profile) + 1)
            plt.plot(
                iterations, 
                profile, 
                alpha=0.3, 
                label=f'Run {i+1}' if i < 5 else None
            )
            
        # Plot média
        avg_profile = self._calculate_average_profile()
        if avg_profile:
            plt.plot(
                range(1, len(avg_profile) + 1),
                avg_profile,
                'r-',
                linewidth=2,
                label='Average'
            )
            
        plt.title('Convergence Analysis')
        plt.xlabel('Improvement Iteration')
        plt.ylabel('Cumulative Improvement')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(save_path)
        plt.close()
        
    def _calculate_average_profile(self) -> List[float]:
        """Calcula perfil médio de convergência."""
        if not self.executions:
            return []
            
        # Encontra comprimento máximo
        max_len = max(
            len(ex['convergence_profile']) 
            for ex in self.executions
        )
        
        # Normaliza perfis
        normalized_profiles = []
        for ex in self.executions:
            profile = ex['convergence_profile']
            if len(profile) < max_len:
                # Estende perfil com último valor
                profile.extend([profile[-1]] * (max_len - len(profile)))
            normalized_profiles.append(profile)
            
        # Calcula média
        return np.mean(normalized_profiles, axis=0).tolist()
        
    def export_analysis(self, output_dir: Path) -> None:
        """Exporta análise completa."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        # Análises
        convergence_stats = self.analyze_convergence()
        neighborhood_stats = self.analyze_neighborhood_effectiveness()
        
        # Prepara relatório
        report = {
            'timestamp': timestamp,
            'num_executions': len(self.executions),
            'best_global_score': float(self.best_global_score),
            'convergence_analysis': convergence_stats,
            'neighborhood_analysis': neighborhood_stats,
            'execution_summaries': [
                {
                    'initial_score': ex['initial_score'],
                    'final_score': ex['final_score'],
                    'num_improvements': len(ex['improvement_path']),
                    'metadata': ex['metadata']
                }
                for ex in self.executions
            ]
        }
        
        # Salva relatório
        report_path = output_dir / f"analysis_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Gera visualizações
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        self.visualize_convergence(
            viz_dir / f"convergence_plot_{timestamp}.png"
        )
        
        self.logger.info(f"Analysis exported to {output_dir}")
        
    def get_stats(self) -> Dict:
        """Retorna estatísticas resumidas para logging."""
        if not self.executions:
            return {}
            
        stats = {
            'num_executions': len(self.executions),
            'best_score': self.best_global_score,
            'avg_score': np.mean([ex['final_score'] for ex in self.executions]),
            'std_score': np.std([ex['final_score'] for ex in self.executions]),
            'avg_improvements': np.mean([
                len(ex['improvement_path']) for ex in self.executions
            ])
        }
        
        return stats
        
    def reset(self) -> None:
        """Reseta analisador."""
        self.executions = []
        self.best_global_score = float('-inf')
        self.best_global_matrix = None
        self.convergence_data.clear()
        self.logger.info("Analyzer reset")