#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
from datetime import datetime

from memetic.matrix import AdaptiveMatrix
from memetic.local_search import LocalSearch
from memetic.clustalw import run_clustalw
from memetic.baliscore import get_bali_score

# Hiperparâmetros do VNS-ILS
HYPERPARAMS = {
    # Parâmetros da busca
    'VNS_MIN_IMPROVEMENT': 1e-6,
    'VNS_MAX_ITER': 20,
    'VNS_MAX_NO_IMPROVE': 10,
    'VNS_PERTURBATION_SIZE': 2,
    'VNS_MAX_PERTURBATION': 5,
    
    # Restrições da matriz PAM
    'MATRIX_SCORE_DIAGONAL': {'min': -2, 'max': 17},
    'MATRIX_SCORE_SIMILAR': {'min': -4, 'max': 8},
    'MATRIX_SCORE_DIFFERENT': {'min': -8, 'max': 4},
    
    # Parâmetros de avaliação 
    'EVAL_SAMPLES': 1,          # Número de avaliações por matriz
    'NUM_RUNS': 3              # Número de execuções independentes
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vns_ils.log'),
            logging.StreamHandler()
        ]
    )

def evaluate_matrix(matrix: AdaptiveMatrix, xml_file: Path, fasta_file: Path) -> float:
    """Avalia uma matriz usando ClustalW e bali_score"""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    matrix_file = temp_dir / "temp_matrix.mat"
    matrix.to_clustalw_format(matrix_file)
    
    scores = []
    for i in range(HYPERPARAMS['EVAL_SAMPLES']):
        aln_file = temp_dir / f"temp_aln_{i}.fasta"
        if run_clustalw(str(fasta_file), str(aln_file), str(matrix_file)):
            score = get_bali_score(str(xml_file), str(aln_file))
            if score > 0:
                scores.append(score)
        
        if aln_file.exists():
            aln_file.unlink()
            
    matrix_file.unlink()
    return sum(scores)/len(scores) if scores else 0.0

def main():
    setup_logging()
    
    # Verifica arquivos de entrada
    project_root = Path(__file__).parent
    xml_file = project_root / "BAliBASE/RV100/BBA0142.xml"
    fasta_file = project_root / "BAliBASE/RV100/BBA0142.tfa"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    if not all(f.exists() for f in [xml_file, fasta_file]):
        logging.error("Arquivos necessários não encontrados")
        sys.exit(1)
        
    logging.info("Iniciando otimização VNS-ILS")
    
    # Executa VNS-ILS múltiplas vezes
    best_overall_score = float('-inf')
    best_overall_matrix = None
    
    for run in range(HYPERPARAMS['NUM_RUNS']):
        logging.info(f"\nIniciando execução {run + 1}/{HYPERPARAMS['NUM_RUNS']}")
        
        try:
            # Inicializa nova busca
            matrix = AdaptiveMatrix()
            local_search = LocalSearch(
                matrix=matrix,
                min_improvement=HYPERPARAMS['VNS_MIN_IMPROVEMENT'],
                perturbation_size=HYPERPARAMS['VNS_PERTURBATION_SIZE'],
                max_perturbation=HYPERPARAMS['VNS_MAX_PERTURBATION'],
                score_constraints={
                    'diagonal': HYPERPARAMS['MATRIX_SCORE_DIAGONAL'],
                    'similar': HYPERPARAMS['MATRIX_SCORE_SIMILAR'],
                    'different': HYPERPARAMS['MATRIX_SCORE_DIFFERENT']
                }
            )
            
            # Analisa alinhamento
            local_search.analyze_alignment(xml_file)
            
            # Executa busca
            score = local_search.vns_search(
                evaluation_func=lambda m: evaluate_matrix(m, xml_file, fasta_file),
                max_iterations=HYPERPARAMS['VNS_MAX_ITER'],
                max_no_improve=HYPERPARAMS['VNS_MAX_NO_IMPROVE']
            )
            
            logging.info(f"Execução {run + 1} finalizada. Score: {score:.4f}")
            
            # Atualiza melhor global
            if score > best_overall_score:
                best_overall_score = score
                best_overall_matrix = local_search.best_matrix
                logging.info(f"Novo melhor score global: {best_overall_score:.4f}")
            
        except Exception as e:
            logging.error(f"Erro durante execução {run + 1}: {str(e)}")
            continue
    
    # Salva melhor resultado
    if best_overall_matrix:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_path = results_dir / f"{timestamp}-VNS-ILS-{best_overall_score:.4f}.txt"
        best_overall_matrix.to_clustalw_format(output_path)
        logging.info(f"Melhor matriz salva em: {output_path}")
    
    logging.info(f"Otimização concluída. Melhor score: {best_overall_score:.4f}")

if __name__ == "__main__":
    main()