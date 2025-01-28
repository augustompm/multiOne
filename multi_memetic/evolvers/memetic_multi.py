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
from multi_memetic.utils.xml_parser import ScoreAccessLayer
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
        self.current_generation = 0  # Adiciona contador de gerações

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
        """Execução do algoritmo memético."""
        self.initial_score = self.population.individuals[0].fitness
        self.best_global_score = self.initial_score
        self.best_global_manager = self.population.individuals[0].matrix_manager.copy()
        stagnation_counter = 0

        while stagnation_counter < max_no_improve:
            if generations % local_search_frequency == 0:
                for ind in self.population.individuals:
                    if ind.local_search_count < 3:
                        # Aplica busca local para cada nível EXPLICITAMENTE
                        for level in ['HIGH', 'MEDIUM', 'LOW']:
                            self.local_search.manager = ind.matrix_manager
                            new_score = self.local_search.vns_search(
                                evaluation_func=evaluation_function,
                                max_iterations=local_search_iterations,
                                max_no_improve=max_no_improve,
                                conservation_level=level  # Passa o nível explicitamente
                            )
                            if new_score > ind.fitness:
                                ind.fitness = new_score
                                ind.matrix_manager.matrices[level] = self.local_search.best_manager.copy()
                                ind.local_search_count += 1

                                if new_score > self.best_global_score:
                                    self.best_global_score = new_score
                                    self.best_global_manager = ind.matrix_manager.copy()
                                    stagnation_counter = 0
                                    continue

            self.population.hierarchical_crossover()

            current_best = self.population.individuals[0]
            if current_best.fitness > self.best_global_score:
                self.best_global_score = current_best.fitness
                self.best_global_manager = current_best.matrix_manager.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            generations -= 1
            if generations <= 0:
                break

        return self.best_global_manager, self.best_global_score

    def run_generations(self, num_generations: int) -> MatrixManager:
        """Executa um número específico de gerações"""
        self.current_generation += num_generations
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