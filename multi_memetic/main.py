# multi_memetic/main.py

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
import xml.etree.ElementTree as ET

# Ajusta o PYTHONPATH corretamente
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Ajusta os imports para a estrutura de pacote
from multi_memetic.evolvers.memetic_multi import MemeticAlgorithmMulti
from multi_memetic.evolvers.population_multi import StructuredPopulationMulti
from multi_memetic.utils.xml_parser import ScoreAccessLayer, ConservationLevel
from multi_memetic.adaptive_matrices.matrix_manager import MatrixManager
from memetic.baliscore import get_bali_score
from memetic.clustalw import run_clustalw

# Instâncias conforme solicitado
INSTANCES = [
    'BBA0005', 'BBA0014', 'BBA0019', 'BBA0022', 
    'BBA0080', 'BBA0126', 'BBA0142', 'BBA0155',
    'BBA0183', 'BBA0185', 'BBA0192', 'BBA0201'
]

# Hyperparâmetros adaptados para multi-matriz
HYPERPARAMS = {
    'VNS': {
        'MIN_IMPROVEMENT': 1e-4,
        'MAX_ITER': 10,
        'MAX_NO_IMPROVE': 5,
        'PERTURBATION_SIZE': 5,
        'NEIGHBORHOOD_THRESHOLDS': {
            'HIGH': [30.0, 25.0],    # Blocos altamente conservados
            'MEDIUM': [22.0, 20.0],  # Blocos média conservação
            'LOW': [15.0, 10.0]      # Blocos baixa conservação
        },
        'MAX_ADJUSTMENTS': {
            'HIGH': [2, 3],          # Ajustes mais restritos
            'MEDIUM': [3, 4],        # Ajustes intermediários
            'LOW': [4, 5]            # Ajustes mais flexíveis
        }
    },
    'MEMETIC': {
        'MAX_GENERATIONS': 50,
        'LOCAL_SEARCH_FREQ': 10,
        'MUTATION_RATE': 0.15,
        'POPULATION_SIZE': 13,
        'HIERARCHY_LEVELS': 3
    },
    'MATRIX': {
        'HIGH': {
            'SCORE_DIAGONAL': {'min': -2, 'max': 17},
            'SCORE_SIMILAR': {'min': -4, 'max': 8},
            'SCORE_DIFFERENT': {'min': -8, 'max': 4},
            'MAX_ADJUSTMENT': 2
        },
        'MEDIUM': {
            'SCORE_DIAGONAL': {'min': -2, 'max': 15},
            'SCORE_SIMILAR': {'min': -3, 'max': 7},
            'SCORE_DIFFERENT': {'min': -6, 'max': 3},
            'MAX_ADJUSTMENT': 3
        },
        'LOW': {
            'SCORE_DIAGONAL': {'min': -2, 'max': 13},
            'SCORE_SIMILAR': {'min': -2, 'max': 6},
            'SCORE_DIFFERENT': {'min': -4, 'max': 3},
            'MAX_ADJUSTMENT': 4
        }
    },
    'EXECUTION': {
        'MATRICES_PER_INSTANCE': 1,
        'EVAL_SAMPLES': 1,
        'MAX_TIME': 300,
        'SEED': None
    }
}

@dataclass
class BestResult:
    matrix_manager: MatrixManager
    score: float
    instance: str
    execution_id: int
    timestamp: str
    hyperparams: Dict
    start_time: float

    def save(self, results_dir: Path) -> Dict[str, Path]:
        """Salva matrizes e meta-dados"""
        execution_time = time.time() - self.start_time
        
        # Exporta matrizes individuais e combinada
        paths = self.matrix_manager.export_matrices(
            results_dir, 
            f"{self.instance}_{self.execution_id+1}_{self.score:.4f}_{execution_time:.1f}"
        )
        
        # Salva metadados
        meta_path = results_dir / f"meta_{self.instance}_{self.execution_id+1}.json"
        metadata = {
            'instance': self.instance,
            'execution_id': self.execution_id + 1,
            'score': self.score,
            'execution_time': execution_time,
            'timestamp': self.timestamp,
            'hyperparams': self.hyperparams,
            'matrix_stats': self.matrix_manager.get_stats()
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return paths
        
    def __lt__(self, other):
        return self.score < other.score

def setup_logging(log_dir: Path) -> None:
    """Configura sistema de logging"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_file = log_dir / f"multi_memetic_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def evaluate_matrix_manager(
    manager: MatrixManager,
    xml_file: Path,
    fasta_file: Path,
    xml_parser: ScoreAccessLayer
) -> float:
    """Avalia conjunto de matrizes usando ClustalW"""
    temp_dir = current_dir / "temp"  # Modificar para usar path absoluto
    if temp_dir.exists():
        for f in temp_dir.glob("*"):
            try:
                f.unlink()
            except:
                pass
    temp_dir.mkdir(exist_ok=True)
    
    # Exporta matriz combinada temporária
    matrix_file = temp_dir / "temp_matrix.mat"
    manager.export_matrices(temp_dir)  # Confirma que o arquivo foi gerado
    
    # Verifica se o arquivo foi criado corretamente
    if not matrix_file.exists():
        logging.error(f"Matrix file not created at {matrix_file}")
        return 0.0
    
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
    """Processa uma única instância"""
    logging.info(f"\nProcessing instance {instance}")
    
    xml_file = input_dir / f"{instance}.xml"
    fasta_file = input_dir / f"{instance}.tfa"
    
    if not all(f.exists() for f in [xml_file, fasta_file]):
        logging.error(f"Required files not found for instance {instance}")
        return None
    
    # Inicializa parser XML    
    xml_parser = ScoreAccessLayer()
    tree = ET.parse(xml_file)
    xml_parser.load_from_xml(tree.getroot())
        
    instance_best = None
    total_executions = hyperparams['EXECUTION']['MATRICES_PER_INSTANCE']
    
    for execution_id in range(total_executions):
        logging.info(f"Starting execution {execution_id + 1}/{total_executions} for {instance}")
        
        start_time = time.time()
        try:
            memetic = MemeticAlgorithmMulti(
                evaluation_function=lambda m: evaluate_matrix_manager(
                    m, xml_file, fasta_file, xml_parser
                ),
                xml_path=xml_file,
                hyperparams=hyperparams
            )
            
            best_manager, best_score = memetic.run(
                generations=hyperparams['MEMETIC']['MAX_GENERATIONS'],
                local_search_frequency=hyperparams['MEMETIC']['LOCAL_SEARCH_FREQ'],
                local_search_iterations=hyperparams['VNS']['MAX_ITER'],
                max_no_improve=hyperparams['VNS']['MAX_NO_IMPROVE'],
                evaluation_function=lambda m: evaluate_matrix_manager(
                    m, xml_file, fasta_file, xml_parser
                )
            )
            
            execution_time = time.time() - start_time
            
            if instance_best is None or best_score > instance_best.score:
                instance_best = BestResult(
                    matrix_manager=best_manager,
                    score=best_score,
                    instance=instance,
                    execution_id=execution_id,
                    timestamp=datetime.now().isoformat(),
                    hyperparams=hyperparams,
                    start_time=start_time
                )
                
                # Salva todas as matrizes usando o novo método
                paths = instance_best.matrix_manager.export_final_matrices(
                    output_dir=results_dir,
                    instance=instance,
                    execution_id=execution_id,
                    score=best_score,
                    execution_time=execution_time
                )
                
                # Log de uso das matrizes
                stats = instance_best.matrix_manager.get_stats()
                logging.info(
                    f"New best score for {instance}: {best_score:.4f}\n" + 
                    f"Matrix usage: " + 
                    ", ".join(f"{k}: {v}" for k, v in stats['usage_count'].items())
                )
            
            execution_time = time.time() - start_time
            logging.info(f"Execution {execution_id + 1} completed in {execution_time:.1f}s")
            
        except Exception as e:
            logging.error(f"Error processing {instance} execution {execution_id + 1}: {str(e)}", 
                         exc_info=True)
            
    if instance_best and (instance not in best_results or 
                         instance_best.score > best_results[instance].score):
        best_results[instance] = instance_best
        
    return instance_best

def main():
    # Setup paths usando o current_dir ao invés de project_root para os diretórios do multi_memetic
    input_dir = project_root / "BAliBASE/RV100"  # BAliBASE está na raiz do projeto
    results_dir = current_dir / "results"         # results dentro do multi_memetic
    log_dir = current_dir / "logs"               # logs dentro do multi_memetic
    temp_dir = current_dir / "temp"              # temp dentro do multi_memetic
    backup_dir = results_dir / "backup"
    
    # Create directories
    for d in [results_dir, log_dir, temp_dir]:
        d.mkdir(exist_ok=True, parents=True)
    
    # Backup existing results
    if results_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)
        for f in results_dir.glob("*"):
            if f.name != "backup":
                shutil.move(str(f), str(backup_path / f.name))
    
    # Setup logging
    setup_logging(log_dir)
    logging.info("Starting multi-matrix executions with hyperparameters:")
    logging.info(json.dumps(HYPERPARAMS, indent=2))
    
    best_results = {}
    
    # Process instances
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
                logging.info("Matrix usage statistics:")
                for level, count in instance_best.matrix_manager.usage_count.items():
                    logging.info(f"  {level}: {count} uses")
        except Exception as e:
            logging.error(f"Failed to process instance {instance}: {str(e)}", exc_info=True)
    
    # Gera relatório final
    summary_path = results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        instance: {
            'score': result.score,
            'execution_id': result.execution_id,
            'timestamp': result.timestamp,
            'matrix_stats': result.matrix_manager.get_stats()
        }
        for instance, result in best_results.items()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("All instances processed successfully")

    # Log final
    total_time = time.time() - start_time
    logging.info(
        f"Optimization completed in {total_time:.1f}s. "
        f"Initial: {self.initial_score:.4f}, Final: {self.best_global_score:.4f}"
    )

    # Exporta estatísticas finais
    final_stats = {
        'runtime': total_time,
        'generations': generation,
        'initial_score': self.initial_score,
        'final_score': self.best_global_score,
        'matrix_stats': self.best_global_manager.get_stats(),
        'timestamp': datetime.now().isoformat()
    }

    stats_path = Path('results') / 'stats' / \
        f"run_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    stats_path.parent.mkdir(exist_ok=True, parents=True)

    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)

if __name__ == "__main__":
    main()