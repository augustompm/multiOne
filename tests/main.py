#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

from memetic.analysis import ExecutionAnalyzer
from memetic.matrix import AdaptiveMatrix
from memetic.memetic import MemeticAlgorithm

# Hiperparâmetros atualizados para main.py

HYPERPARAMS = {
    'VNS': {
        'MIN_IMPROVEMENT': 1e-6,
        'MAX_ITER': 50,                
        'MAX_NO_IMPROVE': 10,           
        'PERTURBATION_SIZE': 5
    },
    'MEMETIC': {
        'MAX_GENERATIONS': 50,
        'LOCAL_SEARCH_FREQ': 5,
        'MUTATION_RATE': 0.1
    },
    'MATRIX': {
        'SCORE_DIAGONAL': {'min': -2, 'max': 17},
        'SCORE_SIMILAR': {'min': -4, 'max': 8},
        'SCORE_DIFFERENT': {'min': -8, 'max': 4},
        'MAX_ADJUSTMENT': 2
    },
    'LOCAL_SEARCH': {
        # Pesos das diferentes estratégias de vizinhança
        'NEIGHBORHOOD_WEIGHTS': {
            'subfamily': 0.4,    # Foco em padrões específicos de subfamílias
            'disorder': 0.3,     # Regiões desordenadas
            'conservation': 0.2, # Conservação global
            'random': 0.1       # Exploração aleatória
        },
        # Ajustes de scores (minimização PAM)
        'SCORE_ADJUSTMENTS': {
            'strong': -2,   # Mudança forte (alta confiança)
            'medium': -1,   # Mudança média
            'weak': 1       # Mudança fraca (exploração)
        },
        # Configurações de análise de padrões
        'PATTERN_WINDOW': 2,     # Tamanho da janela para análise de padrões
        'MIN_CONSERVATION': 0.1, # Threshold mínimo de conservação
        
        # Número de mudanças por estratégia
        'SUBFAMILY_CHANGES': 3,    # Mudanças por subfamília
        'DISORDER_CHANGES': 3,     # Mudanças em regiões desordenadas
        'CONSERVATION_TOP_N': 5,   # Top N AAs mais conservados
        'RANDOM_CHANGES': 3,       # Mudanças aleatórias
        
        # Flags de controle
        'USE_DISORDER_INFO': True  # Habilita uso de info de regiões desordenadas
    },
    'EXECUTION': {
        'MATRICES_PER_INSTANCE': 5,
        'EVAL_SAMPLES': 1,
        'SEED': None
    }
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