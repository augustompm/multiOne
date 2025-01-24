#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

from memetic.analysis import ExecutionAnalyzer
from memetic.matrix import AdaptiveMatrix
from memetic.memetic import MemeticAlgorithm

# Hiperparâmetros do algoritmo
HYPERPARAMS = {
    # Parâmetros populacionais
    'POPULATION_SIZE': 15,      # Tamanho da população
    'ELITE_SIZE': 3,           # Número de indivíduos elite
    'NUM_GENERATIONS': 10,     # Número de gerações
    'NUM_RUNS': 3,            # Número de execuções independentes
    
    # Parâmetros do VNS-ILS
    'MAX_VNS_ITERATIONS': 10,  # Máximo de iterações VNS
    'MAX_NO_IMPROVE': 5,      # Iterações sem melhoria antes de parar
    'LOCAL_SEARCH_FREQ': 5,    # Frequência de busca local
    
    # Parâmetros de avaliação
    'EVAL_SAMPLES': 2,         # Número de avaliações por matriz
    'MIN_IMPROVEMENT': 1e-6,   # Melhoria mínima considerada
}

# Configuração de logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('adaptive_matrix.log'),
            logging.StreamHandler()
        ]
    )

def evaluate_matrix(matrix: AdaptiveMatrix, xml_file: str, fasta_file: str) -> float:
    """Avalia uma matriz usando Clustalw e bali_score"""
    from clustalw import run_clustalw
    from baliscore import get_bali_score
    
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    matrix_file = temp_dir / "temp_matrix.mat"
    matrix.to_clustalw_format(matrix_file)
    
    scores = []
    for _ in range(HYPERPARAMS['EVAL_SAMPLES']):
        aln_file = temp_dir / f"temp_aln_{_}.fasta"
        
        if run_clustalw(fasta_file, str(aln_file), str(matrix_file)):
            score = get_bali_score(xml_file, str(aln_file))
            if score > 0:
                scores.append(score)
                
        if aln_file.exists():
            aln_file.unlink()
            
    matrix_file.unlink()
    return np.mean(scores) if scores else 0.0

def run_optimization(xml_file: str, fasta_file: str) -> AdaptiveMatrix:
    """Executa uma otimização completa"""
    analyzer = ExecutionAnalyzer()
    best_matrix = None
    best_score = float('-inf')
    
    for run in range(HYPERPARAMS['NUM_RUNS']):
        logging.info(f"\nExecutando run {run+1}/{HYPERPARAMS['NUM_RUNS']}")
        
        try:
            # Inicializa algoritmo memético
            memetic = MemeticAlgorithm(
                population_size=HYPERPARAMS['POPULATION_SIZE'],
                elite_size=HYPERPARAMS['ELITE_SIZE'],
                evaluation_function=lambda m: evaluate_matrix(m, xml_file, fasta_file),
                xml_path=Path(xml_file)
            )
            
            # Executa otimização
            current_matrix = memetic.run(
                generations=HYPERPARAMS['NUM_GENERATIONS'],
                local_search_frequency=HYPERPARAMS['LOCAL_SEARCH_FREQ'],
                local_search_iterations=HYPERPARAMS['MAX_VNS_ITERATIONS'],
                max_no_improve=HYPERPARAMS['MAX_NO_IMPROVE']
            )
            
            # Registra execução
            if memetic.best_global_score > best_score:
                best_score = memetic.best_global_score
                best_matrix = current_matrix
                logging.info(f"Novo melhor global encontrado: {best_score:.4f}")
                
            analyzer.record_execution(
                initial_score=memetic.initial_score,
                final_score=memetic.best_global_score,
                improvements=memetic.local_search.improvements,
                final_matrix=current_matrix.matrix
            )
            
        except Exception as e:
            logging.error(f"Erro durante execução {run+1}: {str(e)}")
            continue
            
    return best_matrix

def main():
    setup_logging()
    
    # Definição de caminhos
    project_root = Path(__file__).parent
    xml_file = str(project_root / "BAliBASE/RV100/BBA0142.xml")
    fasta_file = str(project_root / "BAliBASE/RV100/BBA0142.tfa")
    results_dir = project_root / "memetic/results"
    
    # Verifica arquivos necessários
    if not all(Path(f).exists() for f in [xml_file, fasta_file]):
        logging.error("Arquivos de entrada não encontrados")
        sys.exit(1)
        
    logging.info("Iniciando otimização de matriz adaptativa")
    
    # Executa otimização
    best_matrix = run_optimization(xml_file, fasta_file)
    
    # Salva resultado
    if best_matrix:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_path = results_dir / f"{timestamp}-AdaptivePAM.txt"
        best_matrix.to_clustalw_format(output_path)
        logging.info(f"Melhor matriz salva em: {output_path}")
    
    logging.info("Otimização concluída com sucesso")

if __name__ == "__main__":
    main()