#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, Optional

from memetic.matrix import AdaptiveMatrix
from memetic.local_search import LocalSearch
from memetic.memetic import MemeticAlgorithm
from memetic.clustalw import run_clustalw
from memetic.baliscore import get_bali_score

# Configuração dos hiperparâmetros da metaheurística
HYPERPARAMS = {
    # Parâmetros do VNS
    'VNS': {
        'MIN_IMPROVEMENT': 1e-6,        # Mínima melhoria aceita
        'MAX_ITER': 100,                # Máximo de iterações
        'MAX_NO_IMPROVE': 20,           # Máximo de iterações sem melhora
        'PERTURBATION_SIZE': 5,         # Tamanho inicial da perturbação
        'MAX_PERTURBATION': 20,         # Tamanho máximo da perturbação
        'ESCAPE_THRESHOLD': 10,         # Ativa mecanismo de escape após N iterações
        'SCORE_CONSTRAINTS': {  
            'DIAGONAL': {'min': -2, 'max': 17},
            'SIMILAR': {'min': -4, 'max': 8},
            'DIFFERENT': {'min': -8, 'max': 4}
        }
    },
    
    # Parâmetros do algoritmo memético
    'MEMETIC': {
        'POPULATION_SIZE': 13,          # Tamanho da população
        'ELITE_SIZE': 5,                # Tamanho da elite
        'MAX_GENERATIONS': 50,         # Número máximo de gerações
        'LOCAL_SEARCH_FREQ': 5,         # Frequência de busca local
        'DIVERSITY_THRESHOLD': 0.2,     # Limiar de diversidade na elite
    },
    
    # Restrições da matriz PAM
    'MATRIX': {
        'SCORE_DIAGONAL': {'min': -2, 'max': 17},     # Scores na diagonal
        'SCORE_SIMILAR': {'min': -4, 'max': 8},       # Scores entre AAs similares
        'SCORE_DIFFERENT': {'min': -8, 'max': 4},     # Scores entre AAs diferentes
        'MAX_ADJUSTMENT': 2,                          # Máximo ajuste por iteração
    },
    
    # Parâmetros de execução
    'EXECUTION': {
        'NUM_RUNS': 3,                  # Número de execuções independentes
        'EVAL_SAMPLES': 1,              # Avaliações por matriz
        'SEED': None,                   # Seed para reprodutibilidade
    }
}

def setup_logging(log_dir: Path) -> None:
    """Configura sistema de logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_file = log_dir / f"vns_ils_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log dos hiperparâmetros
    logging.info("Iniciando execução com hiperparâmetros:")
    logging.info(json.dumps(HYPERPARAMS, indent=2))

def evaluate_matrix(matrix: AdaptiveMatrix, xml_file: Path, fasta_file: Path) -> float:
    """Avalia uma matriz usando ClustalW e bali_score."""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    matrix_file = temp_dir / "temp_matrix.mat"
    matrix.to_clustalw_format(matrix_file)
    
    scores = []
    for _ in range(HYPERPARAMS['EXECUTION']['EVAL_SAMPLES']):
        aln_file = temp_dir / f"temp_aln_{_}.fasta"
        try:
            if run_clustalw(str(fasta_file), str(aln_file), str(matrix_file)):
                score = get_bali_score(str(xml_file), str(aln_file))
                if score > 0:
                    scores.append(score)
                    logging.debug(f"Matrix evaluation score: {score:.4f}")
        except Exception as e:
            logging.error(f"Error during matrix evaluation: {str(e)}")
        finally:
            if aln_file.exists():
                aln_file.unlink()
    
    if matrix_file.exists():
        matrix_file.unlink()
        
    return sum(scores)/len(scores) if scores else 0.0

def save_results(results_dir: Path, 
                matrix: AdaptiveMatrix, 
                score: float,
                run_info: Optional[Dict] = None) -> None:
    """Salva matriz e informações da execução."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    matrix_file = results_dir / f"AdaptivePAM_{timestamp}_{score:.4f}.txt"
    info_file = results_dir / f"run_info_{timestamp}.json"
    
    try:
        # Salva matriz
        matrix.to_clustalw_format(matrix_file)
        logging.info(f"Matrix saved to: {matrix_file}")
        
        # Salva informações da execução
        if run_info:
            with open(info_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'score': score,
                    'hyperparams': HYPERPARAMS,
                    'run_info': run_info
                }, f, indent=2)
            logging.info(f"Run info saved to: {info_file}")
            
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")

def main():
    # Setup de diretórios
    project_root = Path(__file__).parent
    input_dir = project_root / "BAliBASE/RV100"
    results_dir = project_root / "results"
    log_dir = project_root / "logs"
    
    for d in [results_dir, log_dir]:
        d.mkdir(exist_ok=True)
    
    # Setup do logging
    setup_logging(log_dir)
    
    # Verifica arquivos de entrada
    xml_file = input_dir / "BBA0142.xml"
    fasta_file = input_dir / "BBA0142.tfa"
    
    if not all(f.exists() for f in [xml_file, fasta_file]):
        logging.error("Required input files not found")
        sys.exit(1)
    
    # Inicializa analisador de execuções
    # Removido: ExecutionAnalyzer que gera gráficos
    
    # Executa múltiplas vezes
    best_overall_score = float('-inf')
    best_overall_matrix = None
    
    for run in range(HYPERPARAMS['EXECUTION']['NUM_RUNS']):
        logging.info(f"\nStarting run {run + 1}/{HYPERPARAMS['EXECUTION']['NUM_RUNS']}")
        
        try:
            # Inicializa algoritmo memético
            memetic = MemeticAlgorithm(
                population_size=HYPERPARAMS['MEMETIC']['POPULATION_SIZE'],
                elite_size=HYPERPARAMS['MEMETIC']['ELITE_SIZE'],
                evaluation_function=lambda m: evaluate_matrix(m, xml_file, fasta_file),
                xml_path=xml_file,
                max_generations=HYPERPARAMS['MEMETIC']['MAX_GENERATIONS'],
                local_search_frequency=HYPERPARAMS['MEMETIC']['LOCAL_SEARCH_FREQ'],
                hyperparams=HYPERPARAMS  
            )
            
            # Executa otimização
            best_matrix = memetic.run(
                generations=HYPERPARAMS['MEMETIC']['MAX_GENERATIONS'],
                local_search_frequency=HYPERPARAMS['MEMETIC']['LOCAL_SEARCH_FREQ'],
                local_search_iterations=HYPERPARAMS['VNS']['MAX_ITER'],
                max_no_improve=HYPERPARAMS['VNS']['MAX_NO_IMPROVE']
            )
            
            # Avalia melhor matriz
            final_score = evaluate_matrix(best_matrix, xml_file, fasta_file)
            logging.info(f"Run {run + 1} completed. Score: {final_score:.4f}")
            
            # Atualiza melhor global
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_matrix = best_matrix
                logging.info(f"New best global score: {best_overall_score:.4f}")
            
            # Removido: Registro de execução para análise (ExecutionAnalyzer)
            
        except Exception as e:
            logging.error(f"Error during run {run + 1}: {str(e)}", exc_info=True)
            continue
    
    # Removido: Análise final com ExecutionAnalyzer
    
    # Salva melhor resultado
    if best_overall_matrix:
        save_results(
            results_dir,
            best_overall_matrix,
            best_overall_score,
            {
                'num_runs': HYPERPARAMS['EXECUTION']['NUM_RUNS'],
                'best_score': best_overall_score
                # Removido: 'analyzer_stats': analyzer.get_stats()
            }
        )
  
    logging.info(f"Optimization completed. Best score: {best_overall_score:.4f}")

if __name__ == "__main__":
    main()
