#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
from datetime import datetime
import json
import time
from dataclasses import dataclass
import shutil
import random
import numpy as np
from typing import Dict, Optional, Tuple, List, Set
import xml.etree.ElementTree as ET
from collections import defaultdict

from memetic.baliscore import get_bali_score
from memetic.clustalw import run_clustalw

from multi_memetic.evolvers.memetic_multi import MemeticAlgorithmMulti
from multi_memetic.utils.xml_parser import ScoreAccessLayer, ConservationLevel
from multi_memetic.adaptive_matrices.matrix_manager import MatrixManager

# Instâncias do Reference Set 10
INSTANCES = [
    'BBA0005', 'BBA0014', 'BBA0019', 'BBA0022',
    'BBA0080', 'BBA0126', 'BBA0142', 'BBA0155',
    'BBA0183', 'BBA0185', 'BBA0192', 'BBA0201'
]

HYPERPARAMS = {
    'VNS': {
        'MIN_IMPROVEMENT': 1e-4,
        'MAX_ITER': 10,
        'MAX_NO_IMPROVE': 5,
        'PERTURBATION_SIZE': 5,
        'NEIGHBORHOOD_THRESHOLDS': {
            'HIGH': [30.0, 25.0],
            'MEDIUM': [22.0, 20.0],
            'LOW': [15.0, 10.0]
        }
    },
    'MEMETIC': {
        'MAX_GENERATIONS': 50,
        'LOCAL_SEARCH_FREQ': 10,
        'MUTATION_RATE': 0.15,
        'POPULATION_SIZE': 13,
        'HIERARCHY_LEVELS': 3
    },
    'MATRIX': {  # Adiciona seção MATRIX
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
        'BATCH_SIZE': 3,
        'EVAL_SAMPLES': 1,
        'MAX_TIME': 7200,
        'CHECKPOINT_FREQ': 10,
        'VALIDATION_SPLIT': 0.2  # 20% para validação
    }
}

@dataclass
class BestResult:
    """Mantém resultado da otimização com validação"""
    matrix_manager: MatrixManager
    train_score: float
    validation_score: float
    timestamp: str
    hyperparams: Dict
    stats: Dict

    def save(self, results_dir: Path) -> Dict[str, Path]:
        paths = self.matrix_manager.export_matrices(results_dir)

        meta_path = results_dir / f"meta_{self.timestamp}.json"
        metadata = {
            'train_score': self.train_score,
            'validation_score': self.validation_score,
            'timestamp': self.timestamp,
            'hyperparams': self.hyperparams,
            'stats': self.stats
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return paths

class ReferenceSetAnalyzer:
    """Analisa características globais do Reference Set 10"""

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conservation_stats = defaultdict(list)
        self.block_patterns = defaultdict(int)
        self.sequence_groups = defaultdict(set)

    def analyze(self) -> Dict:
        """Análise completa do Reference Set"""
        for instance in INSTANCES:
            xml_path = self.input_dir / f"{instance}.xml"
            if not xml_path.exists():
                continue

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Analisa blocos conservados
                self._analyze_blocks(root, instance)

                # Analisa grupos de sequências
                self._analyze_sequences(root, instance)

                # Analisa scores de conservação
                self._analyze_conservation(root, instance)

            except Exception as e:
                self.logger.error(f"Error analyzing {instance}: {e}")
                continue

        return self._generate_report()

    def _analyze_blocks(self, root: ET.Element, instance: str) -> None:
        """Análise detalhada dos blocos conservados"""
        for block in root.findall(".//fitem/[ftype='BLOCK']"):
            try:
                score = float(block.find("fscore").text)
                length = int(block.find("fstop").text) - int(block.find("fstart").text) + 1

                if score > 25.0:
                    level = ConservationLevel.HIGH
                elif 20.0 <= score <= 25.0:
                    level = ConservationLevel.MEDIUM
                else:
                    level = ConservationLevel.LOW

                self.conservation_stats[level].append({
                    'instance': instance,
                    'score': score,
                    'length': length
                })

            except Exception:
                continue

    def _analyze_sequences(self, root: ET.Element, instance: str) -> None:
        """Análise dos grupos de sequências"""
        for seq in root.findall(".//sequence"):
            try:
                seq_name = seq.find("seq-name").text
                group = int(seq.find("seq-info/group").text)
                self.sequence_groups[instance].add(group)
            except Exception:
                continue

    def _analyze_conservation(self, root: ET.Element, instance: str) -> None:
        """Análise dos padrões de conservação"""
        for col_score in root.findall(".//column-score"):
            try:
                name = col_score.find("colsco-name").text
                if 'normd' in name.lower():
                    data = [int(x) for x in col_score.find("colsco-data").text.split()]
                    self.conservation_stats[instance].extend(data)
            except Exception:
                continue

    def _generate_report(self) -> Dict:
        """Gera relatório completo da análise"""
        report = {
            'conservation_levels': {},
            'sequence_groups': {},
            'recommendations': {}
        }

        # Estatísticas por nível
        for level in [ConservationLevel.HIGH, ConservationLevel.MEDIUM, ConservationLevel.LOW]:
            blocks = self.conservation_stats[level]
            if blocks:
                report['conservation_levels'][level] = {
                    'count': len(blocks),
                    'avg_score': np.mean([b['score'] for b in blocks]),
                    'avg_length': np.mean([b['length'] for b in blocks]),
                    'instances': len(set(b['instance'] for b in blocks))
                }

        # Análise de grupos
        for instance, groups in self.sequence_groups.items():
            report['sequence_groups'][instance] = len(groups)

        # Recomendações para otimização
        report['recommendations'] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> Dict:
        """Gera recomendações para otimização"""
        recs = {}

        # Para cada nível de conservação
        for level in [ConservationLevel.HIGH, ConservationLevel.MEDIUM, ConservationLevel.LOW]:
            blocks = self.conservation_stats[level]
            if not blocks:
                continue

            scores = [b['score'] for b in blocks]
            lengths = [b['length'] for b in blocks]

            recs[level] = {
                'target_score_range': (np.percentile(scores, 25), np.percentile(scores, 75)),
                'typical_block_length': np.median(lengths),
                'score_weight': len(blocks) / sum(len(self.conservation_stats[l]) 
                                                  for l in self.conservation_stats)
            }

        return recs

class MatrixEvaluator:
    """Sistema de avaliação de matrizes com validação"""

    def __init__(
        self,
        input_dir: Path,
        instances: List[str],
        hyperparams: Dict,
        xml_parser: ScoreAccessLayer
    ):
        self.input_dir = input_dir
        self.instances = instances.copy()
        self.hyperparams = hyperparams
        self.xml_parser = xml_parser
        self.logger = logging.getLogger(self.__class__.__name__)

        # Separa conjunto de validação
        random.shuffle(self.instances)
        split = int(len(self.instances) * hyperparams['EXECUTION']['VALIDATION_SPLIT'])
        self.validation_instances = self.instances[:split]
        self.training_instances = self.instances[split:]

        self.logger.info(
            f"Split instances - Training: {len(self.training_instances)}, "
            f"Validation: {len(self.validation_instances)}"
        )

    def evaluate_training(
        self,
        manager: MatrixManager,
        batch_size: Optional[int] = None
    ) -> Tuple[float, Dict]:
        """Avalia em instâncias de treino"""
        return self._evaluate_instances(
            manager,
            self.training_instances,
            batch_size
        )

    def evaluate_validation(self, manager: MatrixManager) -> Tuple[float, Dict]:
        """Avalia em instâncias de validação"""
        return self._evaluate_instances(
            manager,
            self.validation_instances
        )

    def _evaluate_instances(
        self,
        manager: MatrixManager,
        instances: List[str],
        batch_size: Optional[int] = None
    ) -> Tuple[float, Dict]:
        """Avalia conjunto de instâncias"""
        if batch_size:
            # Amostra aleatória se batch_size definido
            if len(instances) > batch_size:
                instances = random.sample(instances, batch_size)

        scores = []
        stats = defaultdict(list)

        for instance in instances:
            xml_file = self.input_dir / f"{instance}.xml"
            fasta_file = self.input_dir / f"{instance}.tfa"

            if not all(f.exists() for f in [xml_file, fasta_file]):
                continue

            # Avalia com múltiplas amostras
            instance_scores = []
            for _ in range(self.hyperparams['EXECUTION']['EVAL_SAMPLES']):
                score = self._evaluate_single(
                    manager,
                    xml_file,
                    fasta_file
                )
                if score > 0:
                    instance_scores.append(score)

            if instance_scores:
                avg_score = np.mean(instance_scores)
                scores.append(avg_score)
                stats[instance].extend(instance_scores)

        if not scores:
            return 0.0, {}

        return np.mean(scores), {
            'min': min(scores),
            'max': max(scores),
            'std': np.std(scores),
            'instances': {
                inst: {
                    'mean': np.mean(s),
                    'std': np.std(s)
                }
                for inst, s in stats.items()
            }
        }

    def _evaluate_single(
        self,
        manager: MatrixManager,
        xml_file: Path,
        fasta_file: Path
    ) -> float:
        """Avalia uma única instância"""
        temp_dir = Path("multi_memetic/temp")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Exporta todas as matrizes
            paths = manager.export_matrices(temp_dir)
            
            # Usa a matriz combinada para avaliação
            matrix_path = paths['combined']
            aln_file = temp_dir / "temp_aln.fasta"

            if run_clustalw(str(fasta_file), str(aln_file), str(matrix_path)):
                return get_bali_score(str(xml_file), str(aln_file))

        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")

        finally:
            # Limpa arquivos temporários
            for f in paths.values():
                if f.exists():
                    f.unlink()
            if 'aln_file' in locals() and aln_file.exists():
                aln_file.unlink()

        return 0.0

def setup_environment() -> Tuple[Path, Path, Path]:
    """Prepara ambiente de execução"""
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "BAliBASE/RV100"
    results_dir = Path("multi_memetic/results")
    backup_dir = results_dir / "backup"
    log_dir = Path("multi_memetic/logs")

    # Cria diretórios
    for d in [results_dir, backup_dir, log_dir]:
        d.mkdir(exist_ok=True, parents=True)

    # Backup de resultados anteriores
    if results_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / timestamp
        backup_path.mkdir(exist_ok=True)

        for f in results_dir.glob("*"):
            if f.name != "backup":
                shutil.move(str(f), str(backup_path / f.name))

    return input_dir, results_dir, log_dir

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

def main():
    # Setup inicial
    input_dir, results_dir, log_dir = setup_environment()
    setup_logging(log_dir)
    start_time = time.time()

    # Análise do Reference Set
    logging.info("Starting Reference Set analysis...")
    analyzer = ReferenceSetAnalyzer(input_dir)
    analysis = analyzer.analyze()

    logging.info("Reference Set analysis complete:")
    logging.info(json.dumps(analysis, indent=2))

    # Salva análise
    analysis_file = results_dir / "reference_set_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Prepara avaliador
    xml_parser = ScoreAccessLayer()
    evaluator = MatrixEvaluator(
        input_dir=input_dir,
        instances=INSTANCES,
        hyperparams=HYPERPARAMS,
        xml_parser=xml_parser
    )

    # 1. Primeiro analisa todo o Reference Set
    reference_set_analysis = analyzer.analyze()  # ...existing code...

    # Inicializa algoritmo memético
    memetic = MemeticAlgorithmMulti(
        evaluation_function=lambda m: evaluator.evaluate_training(
            m, HYPERPARAMS['EXECUTION']['BATCH_SIZE']
        )[0],
        hyperparams=HYPERPARAMS,
        reference_analysis=analysis  # Usa análise do Reference Set
    )

    # Processo evolutivo principal
    best_validation_score = float('-inf')
    generations_no_improve = 0
    checkpoints = []

    logging.info("Starting evolutionary process...")

    try:
        while (time.time() - start_time < HYPERPARAMS['EXECUTION']['MAX_TIME'] and 
               generations_no_improve < HYPERPARAMS['MEMETIC']['MAX_GENERATIONS']):

            # Evolução por algumas gerações
            current_manager = memetic.run_generations(
                HYPERPARAMS['EXECUTION']['CHECKPOINT_FREQ']
            )

            # Avalia em validação
            val_score, val_stats = evaluator.evaluate_validation(current_manager)

            logging.info(
                f"Generation {memetic.current_generation}: "
                f"Training={memetic.best_global_score:.4f}, "
                f"Validation={val_score:.4f}"
            )

            if val_score > best_validation_score:
                best_validation_score = val_score
                generations_no_improve = 0
                
                # Salva checkpoint
                checkpoint = {
                    'generation': memetic.current_generation,
                    'matrices': current_manager.copy(),
                    'train_score': memetic.best_global_score,
                    'validation_score': val_score,
                    'timestamp': datetime.now().isoformat()
                }
                checkpoints.append(checkpoint)
                checkpoints.sort(key=lambda x: x['validation_score'], reverse=True)
                checkpoints = checkpoints[:3]
            else:
                generations_no_improve += HYPERPARAMS['EXECUTION']['CHECKPOINT_FREQ']
                
    except Exception as e:
        logging.error(f"Error in optimization: {str(e)}")

    # Processo finalizado - seleciona melhor resultado
    if checkpoints:
        best_checkpoint = checkpoints[0]
        final_result = BestResult(
            matrix_manager=best_checkpoint['matrices'],
            train_score=best_checkpoint['train_score'],
            validation_score=best_checkpoint['validation_score'],
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            hyperparams=HYPERPARAMS,
            stats={
                'total_time': time.time() - start_time,
                'final_generation': memetic.current_generation,
                'checkpoints': len(checkpoints),
                'matrix_stats': best_checkpoint['matrices'].get_stats()
            }
        )

        # Salva resultado final
        paths = final_result.save(results_dir)

        logging.info("\nOptimization Complete!")
        logging.info(f"Best Training Score: {final_result.train_score:.4f}")
        logging.info(f"Best Validation Score: {final_result.validation_score:.4f}")
        logging.info(f"Total Time: {final_result.stats['total_time']/3600:.1f}h")
        logging.info(f"Final Generation: {final_result.stats['final_generation']}")
        logging.info("\nMatrix Statistics:")
        for level, stats in final_result.stats['matrix_stats'].items():
            logging.info(f"{level}:")
            logging.info(json.dumps(stats, indent=2))
        logging.info(f"\nResults saved to: {paths}")

        # Salva checkpoints para análise
        checkpoint_dir = results_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        for i, ckpt in enumerate(checkpoints):
            ckpt_file = checkpoint_dir / f"checkpoint_{i+1}.json"
            with open(ckpt_file, 'w') as f:
                # Remove matriz do checkpoint para salvar
                ckpt_data = {k: v for k, v in ckpt.items() if k != 'matrices'}
                json.dump(ckpt_data, f, indent=2)
    else:
        logging.warning("No checkpoints were saved. Optimization may not have completed successfully.")

if __name__ == "__main__":
    main()
