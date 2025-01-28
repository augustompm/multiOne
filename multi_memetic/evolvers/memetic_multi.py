# multi_memetic/evolvers/memetic_multi.py

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional, Callable
import random
from datetime import datetime
import time
import xml.etree.ElementTree as ET  # Adicionar esta linha

from multi_memetic.evolvers.population_multi import StructuredPopulationMulti, IndividualMulti
from multi_memetic.utils.xml_parser import ScoreAccessLayer, ConservationLevel
from multi_memetic.adaptive_matrices.matrix_manager import MatrixManager
from multi_memetic.evolvers.vns_ils_multi import VNSILS

class MemeticAlgorithmMulti:
    """
    Algoritmo Memético com população estruturada e VNS-ILS
    adaptado para trabalhar com três matrizes por indivíduo.
    """
    def __init__(
        self,
        evaluation_function: Callable,
        hyperparams: Dict,
        reference_analysis: Dict
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hyperparams = hyperparams
        self.evaluation_function = evaluation_function
        self.reference_analysis = reference_analysis

        # Inicializa busca local VNS
        self.local_search = VNSILS(
            MatrixManager(hyperparams),
            hyperparams
        )

        # População estruturada - ajusta parâmetros passados
        self.population = StructuredPopulationMulti(
            evaluation_function=evaluation_function,
            hyperparams=hyperparams,
            reference_analysis=reference_analysis
        )

        self.best_global_manager = None
        self.best_global_score = float('-inf')
        self.initial_score = None

        # Controle de tempo
        self.start_time = time.time()

    def run(
        self,
        generations: int,
        local_search_frequency: int,
        local_search_iterations: int,
        max_no_improve: int,
        evaluation_function: Callable
    ) -> Tuple[MatrixManager, float]:
        """
        Executa algoritmo memético retornando melhor conjunto de matrizes.
        """
        self.initial_score = self.population.individuals[0].fitness
        self.best_global_score = self.initial_score
        self.best_global_manager = self.population.individuals[0].matrix_manager.copy()
        
        stagnation_counter = 0
        generation = 0

        while (generation < generations and 
               stagnation_counter < max_no_improve and
               time.time() - self.start_time < self.hyperparams['EXECUTION']['MAX_TIME']):

            # Busca local periódica
            if generation % local_search_frequency == 0:
                for ind in self.population.individuals:
                    if ind.local_search_count < 3:  # Limite de aplicações
                        # Busca local em cada matriz do indivíduo
                        for level in [ConservationLevel.HIGH, 
                                    ConservationLevel.MEDIUM,
                                    ConservationLevel.LOW]:
                            matrix = ind.matrix_manager.get_matrix(level)
                            blocks = self.xml_parser.get_blocks_by_conservation(level)
                            
                            # Aplica VNS-ILS na matriz específica
                            self.local_search.matrix = matrix
                            new_score = self.local_search.vns_search(
                                evaluation_func=evaluation_function,
                                max_iterations=local_search_iterations,
                                max_no_improve=max_no_improve,
                                blocks=blocks,
                                conservation_level=level  # Adiciona o nível de conservação
                            )

                            if new_score > ind.fitness:
                                ind.fitness = new_score
                                ind.matrix_manager.matrices[level] = \
                                    self.local_search.best_matrix.copy()
                                ind.local_search_count += 1

                                # Atualiza melhor global se necessário
                                if new_score > self.best_global_score:
                                    self.best_global_score = new_score
                                    self.best_global_manager = \
                                        ind.matrix_manager.copy()
                                    self.logger.info(
                                        f"New best score: {self.best_global_score:.4f}")
                                    stagnation_counter = 0
                                    continue

            # Crossover hierárquico
            self.population.hierarchical_crossover()

            # Verifica melhoria
            current_best = self.population.individuals[0]
            if current_best.fitness > self.best_global_score:
                self.best_global_score = current_best.fitness
                self.best_global_manager = current_best.matrix_manager.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Log periódico
            if generation % 5 == 0:
                elapsed = time.time() - self.start_time
                self.logger.info(
                    f"Generation {generation}: Best={self.best_global_score:.4f}, "
                    f"Time={elapsed:.1f}s, Stagnation={stagnation_counter}"
                )

                # Log de uso das matrizes
                stats = self.best_global_manager.get_stats()
                self.logger.info(
                    "Matrix usage: " + 
                    ", ".join(f"{k}: {v}" for k,v in stats['usage_count'].items())
                )

            generation += 1

        # Log final
        total_time = time.time() - self.start_time
        self.logger.info(
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

        return self.best_global_manager.copy(), self.best_global_score

    def run_generations(self, num_generations: int) -> MatrixManager:
        """Executa um número específico de gerações"""
        return self.run(
            generations=num_generations,
            local_search_frequency=self.hyperparams['MEMETIC']['LOCAL_SEARCH_FREQ'],
            local_search_iterations=self.hyperparams['VNS']['MAX_ITER'],
            max_no_improve=self.hyperparams['VNS']['MAX_NO_IMPROVE'],
            evaluation_function=self.evaluation_function
        )[0]  # Retorna apenas o manager, não o score

    def _check_stagnation(self, stagnation_counter: int, 
                         max_no_improve: int) -> bool:
        """Verifica critério de parada por estagnação"""
        if stagnation_counter >= max_no_improve:
            self.logger.info(
                f"Stopping: No improvement for {max_no_improve} generations")
            return True
        return False

    def _check_time_limit(self) -> bool:
        """Verifica limite de tempo de execução"""
        if time.time() - self.start_time >= self.hyperparams['EXECUTION']['MAX_TIME']:
            self.logger.info("Stopping: Time limit reached")
            return True
        return False