# memetic/analysis.py

import numpy as np
import logging
from typing import List, Dict, Tuple
from pathlib import Path

class ExecutionAnalyzer:
    def __init__(self):
        self.executions = []
        self.best_global_score = float('-inf')
        self.best_global_matrix = None
        
    def record_execution(self, initial_score: float, final_score: float, 
                        improvements: List[Dict], final_matrix: np.ndarray):
        execution = {
            'initial_score': initial_score,
            'final_score': final_score,
            'num_improvements': len(improvements),
            'improvement_path': [imp['new_score'] for imp in improvements],
            'final_matrix': final_matrix.copy()
        }
        self.executions.append(execution)
        
        # Atualizar melhor global
        if final_score > self.best_global_score:
            self.best_global_score = final_score
            self.best_global_matrix = final_matrix.copy()
            logging.info(f"Novo melhor global encontrado: {self.best_global_score:.4f}")
    
    def analyze(self):
        scores = [ex['final_score'] for ex in self.executions]
        
        if not scores:  # Se não tem execuções
            logging.info(f"Análise de 0 execuções: Nenhum dado disponível")
            return
            
        logging.info(f"Análise de {len(self.executions)} execuções:")
        if scores:
            logging.info(f"  Média: {np.mean(scores):.4f}")
            logging.info(f"  Melhor: {max(scores):.4f}")
            logging.info(f"  Pior: {min(scores):.4f}")
        
    def compare_matrices(self, matrix1: np.ndarray, matrix2: np.ndarray, 
                        aa_order: str) -> List[Tuple[str, str, float, float]]:
        """Compara duas matrizes e retorna as diferenças mais significativas."""
        differences = []
        for i, aa1 in enumerate(aa_order):
            for j, aa2 in enumerate(aa_order):
                diff = matrix2[i,j] - matrix1[i,j]
                if abs(diff) > 0:
                    differences.append((aa1, aa2, matrix1[i,j], matrix2[i,j]))
        return sorted(differences, key=lambda x: abs(x[3]-x[2]), reverse=True)
    
    def visualize_execution_path(self, execution_index: int, save_path: Path) -> None:
        """
        Gera um gráfico do caminho de melhorias para uma execução específica.
        
        Args:
            execution_index: Índice da execução a ser visualizada.
            save_path: Caminho onde o gráfico será salvo.
        """
        import matplotlib.pyplot as plt  # Import adicionado para visualização
        
        if execution_index < 0 or execution_index >= len(self.executions):
            logging.error("Índice de execução inválido.")
            return
        
        execution = self.executions[execution_index]
        improvement_path = execution['improvement_path']
        
        plt.figure(figsize=(10, 6))
        plt.plot(improvement_path, marker='o')
        plt.title(f"Caminho de Melhorias - Execução {execution_index + 1}")
        plt.xlabel("Iteração de Melhoria")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Gráfico salvo em {save_path}.")
    
    def export_results_to_csv(self, csv_path: Path) -> None:
        """
        Exporta os resultados das execuções para um arquivo CSV.
        
        Args:
            csv_path: Caminho do arquivo CSV a ser criado.
        """
        import csv
        
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Execução', 'Initial Score', 'Final Score', 
                             'Número de Melhorias'])
            
            for idx, ex in enumerate(self.executions, start=1):
                writer.writerow([
                    idx,
                    ex['initial_score'],
                    ex['final_score'],
                    ex['num_improvements']
                ])
        
        logging.info(f"Resultados exportados para {csv_path}.")
    
    def get_top_n_executions(self, n: int) -> List[Dict]:
        """
        Retorna as top N execuções com os melhores scores finais.
        
        Args:
            n: Número de execuções a serem retornadas.
        
        Returns:
            Lista das top N execuções.
        """
        sorted_executions = sorted(self.executions, key=lambda x: x['final_score'], reverse=True)
        return sorted_executions[:n]
    
    def reset(self):
        """Reseta todas as execuções armazenadas."""
        self.executions = []
        self.best_global_score = float('-inf')
        self.best_global_matrix = None
        logging.info("Todas as execuções e melhores globais foram resetadas.")
