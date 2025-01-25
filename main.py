#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, Optional, Tuple, List
import time
from dataclasses import dataclass

from memetic.matrix import AdaptiveMatrix
from memetic.local_search import LocalSearch
from memetic.memetic import MemeticAlgorithm
from memetic.clustalw import run_clustalw
from memetic.baliscore import get_bali_score

# Lista de instâncias a serem processadas
INSTANCES = [
    'BBA0004', 'BBA0005', 'BBA0008', 'BBA0011', 'BBA0014', 'BBA0015', 
    'BBA0019', 'BBA0021', 'BBA0022', 'BBA0024', 'BBA0080', 'BBA0126', 
    'BBA0133', 'BBA0142', 'BBA0148', 'BBA0155', 'BBA0163', 'BBA0178', 
    'BBA0183', 'BBA0185', 'BBA0192', 'BBA0201', 'BBA0218'
]

# Hiperparâmetros ajustados para instâncias maiores
HYPERPARAMS = {
    'VNS': {
        'MIN_IMPROVEMENT': 1e-6,
        'MAX_ITER': 50,                
        'MAX_NO_IMPROVE': 10,           
        'PERTURBATION_SIZE': 10,        
        'MAX_PERTURBATION': 30,         
        'ESCAPE_THRESHOLD': 15,         
        'SCORE_CONSTRAINTS': {
            'DIAGONAL': {'min': -2, 'max': 17},
            'SIMILAR': {'min': -4, 'max': 8},
            'DIFFERENT': {'min': -8, 'max': 4}
        }
    },
    'MEMETIC': {
        'POPULATION_SIZE': 30,          
        'ELITE_SIZE': 8,                
        'MAX_GENERATIONS': 50,         
        'LOCAL_SEARCH_FREQ': 5,
        'DIVERSITY_THRESHOLD': 0.2,
    },
    'MATRIX': {
        'SCORE_DIAGONAL': {'min': -2, 'max': 17},
        'SCORE_SIMILAR': {'min': -4, 'max': 8},
        'SCORE_DIFFERENT': {'min': -8, 'max': 4},
        'MAX_ADJUSTMENT': 2,
    },
    'EXECUTION': {
        'NUM_RUNS': 1,                  # Fixado em 1
        'MATRICES_PER_INSTANCE': 5,     # Número de matrizes por instância
        'EVAL_SAMPLES': 1,
        'SEED': None,
    }
}

@dataclass
class ExecutionResult:
    matrix: AdaptiveMatrix
    score: float
    execution_time: float

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

def evaluate_matrix(matrix: AdaptiveMatrix, xml_file: Path, fasta_file: Path) -> float:
    """Avalia uma matriz usando ClustalW e bali_score."""
    temp_dir = Path("memetic/temp")
    if temp_dir.exists():
        for f in temp_dir.glob("*"):
            try:
                f.unlink()
            except:
                pass
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
        finally:
            if aln_file.exists():
                aln_file.unlink()
    
    if matrix_file.exists():
        matrix_file.unlink()
        
    return sum(scores)/len(scores) if scores else 0.0

def process_instance(instance: str, input_dir: Path, results_dir: Path) -> List[ExecutionResult]:
    """Processa uma única instância."""
    logging.info(f"\nProcessing instance {instance}")
    
    xml_file = input_dir / f"{instance}.xml"
    fasta_file = input_dir / f"{instance}.tfa"
    
    if not all(f.exists() for f in [xml_file, fasta_file]):
        logging.error(f"Required files not found for instance {instance}")
        return []
        
    results = []
    for execution_id in range(HYPERPARAMS['EXECUTION']['MATRICES_PER_INSTANCE']):
        logging.info(f"Starting execution {execution_id + 1}/5 for {instance}")
        
        start_time = time.time()
        try:
            memetic = MemeticAlgorithm(
                population_size=HYPERPARAMS['MEMETIC']['POPULATION_SIZE'],
                elite_size=HYPERPARAMS['MEMETIC']['ELITE_SIZE'],
                evaluation_function=lambda m: evaluate_matrix(m, xml_file, fasta_file),
                xml_path=xml_file,
                max_generations=HYPERPARAMS['MEMETIC']['MAX_GENERATIONS'],
                local_search_frequency=HYPERPARAMS['MEMETIC']['LOCAL_SEARCH_FREQ'],
                hyperparams=HYPERPARAMS
            )
            
            best_matrix, best_score = memetic.run(
                generations=HYPERPARAMS['MEMETIC']['MAX_GENERATIONS'],
                local_search_frequency=HYPERPARAMS['MEMETIC']['LOCAL_SEARCH_FREQ'],
                local_search_iterations=HYPERPARAMS['VNS']['MAX_ITER'],
                max_no_improve=HYPERPARAMS['VNS']['MAX_NO_IMPROVE'],
                evaluation_function=lambda m: evaluate_matrix(m, xml_file, fasta_file)
            )
            
            execution_time = time.time() - start_time
            
            results.append(ExecutionResult(
                matrix=best_matrix,
                score=best_score,
                execution_time=execution_time
            ))
            
            # Salva matriz
            matrix_file = results_dir / f"{instance},{execution_id + 1},{best_score:.4f},{execution_time:.1f}.txt"
            best_matrix.to_clustalw_format(matrix_file)
            logging.info(f"Matrix saved: {matrix_file}")
            
        except Exception as e:
            logging.error(f"Error processing {instance} execution {execution_id + 1}: {str(e)}", exc_info=True)
            
    return results

def main():
    # Setup de diretórios
    project_root = Path(__file__).parent
    input_dir = project_root / "BAliBASE/RV100"
    results_dir = project_root / "memetic/results"
    log_dir = project_root / "logs"
    temp_dir = project_root / "memetic/temp"
    
    # Cria diretórios necessários
    for d in [results_dir, log_dir, temp_dir]:
        d.mkdir(exist_ok=True, parents=True)
    
    # Setup do logging
    setup_logging(log_dir)
    logging.info("Starting executions with hyperparameters:")
    logging.info(json.dumps(HYPERPARAMS, indent=2))
    
    # Processa cada instância
    all_results = {}
    for instance in INSTANCES:
        instance_results = process_instance(instance, input_dir, results_dir)
        all_results[instance] = instance_results
        
        # Log resultados da instância
        if instance_results:
            best_score = max(r.score for r in instance_results)
            avg_time = sum(r.execution_time for r in instance_results) / len(instance_results)
            logging.info(f"{instance} completed - Best score: {best_score:.4f}, Avg time: {avg_time:.1f}s")
    
    logging.info("All instances processed successfully")

if __name__ == "__main__":
    main()