#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, Optional, Tuple, List
import time
from dataclasses import dataclass
import shutil

from memetic.matrix import AdaptiveMatrix
from memetic.local_search import EnhancedLocalSearch
from memetic.memetic import MemeticAlgorithm
from memetic.clustalw import run_clustalw
from memetic.baliscore import get_bali_score

INSTANCES = [
  'BBA0004'
]

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
        'NEIGHBORHOOD_WEIGHTS': {
            'subfamily': 0.4,
            'disorder': 0.3,
            'conservation': 0.2,
            'random': 0.1
        },
        'SCORE_ADJUSTMENTS': {
            'strong': -2,
            'medium': -1,
            'weak': 1
        },
        'PATTERN_WINDOW': 2,
        'MIN_CONSERVATION': 0.1,
        'SUBFAMILY_CHANGES': 3,
        'DISORDER_CHANGES': 3,
        'CONSERVATION_TOP_N': 5,
        'RANDOM_CHANGES': 3,
        'USE_DISORDER_INFO': True
    },
    'EXECUTION': {
        'MATRICES_PER_INSTANCE': 1,
        'EVAL_SAMPLES': 1,
        'SEED': None
    }
}

@dataclass
class BestResult:
    matrix: AdaptiveMatrix
    score: float
    instance: str
    execution_id: int
    timestamp: str
    hyperparams: Dict
    start_time: float

    def save(self, results_dir: Path) -> Path:
        execution_time = time.time() - self.start_time
        matrix_filename = f"{self.instance},{self.execution_id+1},{self.score:.4f},{execution_time:.1f}.txt"
        matrix_path = results_dir / matrix_filename
        self.matrix.to_clustalw_format(matrix_path)
        
        meta_filename = matrix_filename.replace('.txt', '_meta.json')
        meta_path = results_dir / meta_filename
        metadata = {
            'instance': self.instance,
            'execution_id': self.execution_id + 1,
            'score': self.score,
            'execution_time': execution_time,
            'timestamp': self.timestamp,
            'hyperparams': self.hyperparams
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return matrix_path
        
    def __lt__(self, other):
        return self.score < other.score

def setup_logging(log_dir: Path) -> None:
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

def process_instance(
    instance: str,
    input_dir: Path,
    results_dir: Path,
    hyperparams: Dict,
    best_results: Dict[str, BestResult]
) -> Optional[BestResult]:
    """Processa uma única instância."""
    logging.info(f"\nProcessing instance {instance}")
    
    xml_file = input_dir / f"{instance}.xml"
    fasta_file = input_dir / f"{instance}.tfa"
    
    if not all(f.exists() for f in [xml_file, fasta_file]):
        logging.error(f"Required files not found for instance {instance}")
        return None
        
    instance_best = None
    total_executions = hyperparams['EXECUTION']['MATRICES_PER_INSTANCE']
    
    for execution_id in range(total_executions):
        logging.info(f"Starting execution {execution_id + 1}/{total_executions} for {instance}")
        
        start_time = time.time()
        try:
            memetic = MemeticAlgorithm(
                evaluation_function=lambda m: evaluate_matrix(m, xml_file, fasta_file),
                xml_path=xml_file,
                hyperparams=hyperparams
            )
            
            best_matrix, best_score = memetic.run(
                generations=hyperparams['MEMETIC']['MAX_GENERATIONS'],
                local_search_frequency=hyperparams['MEMETIC']['LOCAL_SEARCH_FREQ'],
                local_search_iterations=hyperparams['VNS']['MAX_ITER'],
                max_no_improve=hyperparams['VNS']['MAX_NO_IMPROVE'],
                evaluation_function=lambda m: evaluate_matrix(m, xml_file, fasta_file)
            )
            
            execution_time = time.time() - start_time
            
            if instance_best is None or best_score > instance_best.score:
                instance_best = BestResult(
                    matrix=best_matrix,
                    score=best_score,
                    instance=instance,
                    execution_id=execution_id,
                    timestamp=datetime.now().isoformat(),
                    hyperparams=hyperparams,
                    start_time=start_time
                )
                instance_best.save(results_dir)
                logging.info(f"New best score for {instance}: {best_score:.4f}")
            
            logging.info(f"Execution {execution_id + 1} completed in {execution_time:.1f}s")
            
        except Exception as e:
            logging.error(f"Error processing {instance} execution {execution_id + 1}: {str(e)}", exc_info=True)
            
    if instance_best and (instance not in best_results or instance_best.score > best_results[instance].score):
        best_results[instance] = instance_best
        
    return instance_best

def main():
    project_root = Path(__file__).parent
    input_dir = project_root / "BAliBASE/RV100"
    results_dir = project_root / "memetic/results"
    log_dir = project_root / "logs"
    temp_dir = project_root / "memetic/temp"
    backup_dir = results_dir / "backup"
    
    for d in [results_dir, log_dir, temp_dir]:
        d.mkdir(exist_ok=True, parents=True)
    
    if results_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)
        for f in results_dir.glob("*"):
            if f.name != "backup":
                shutil.move(str(f), str(backup_path / f.name))
    
    setup_logging(log_dir)
    logging.info("Starting executions with hyperparameters:")
    logging.info(json.dumps(HYPERPARAMS, indent=2))
    
    best_results = {}
    
    for instance in INSTANCES:
        try:
            instance_best = process_instance(
                instance=instance,
                input_dir=input_dir,
                results_dir=results_dir,
                hyperparams=HYPERPARAMS,
                best_results=best_results
            )
            if instance_best:
                logging.info(f"{instance} best score: {instance_best.score:.4f}")
        except Exception as e:
            logging.error(f"Failed to process instance {instance}: {str(e)}", exc_info=True)
    
    summary_path = results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        instance: {
            'score': result.score,
            'execution_id': result.execution_id,
            'timestamp': result.timestamp
        }
        for instance, result in best_results.items()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("All instances processed successfully")

if __name__ == "__main__":
    main()