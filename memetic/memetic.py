import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional, Callable
import random
from dataclasses import dataclass
from datetime import datetime
import time

from .matrix import AdaptiveMatrix
from .local_search import VNSILS  # changed from EnhancedLocalSearch

@dataclass
class Individual:
    matrix: AdaptiveMatrix
    fitness: float = float('-inf')
    local_search_count: int = 0

    def copy(self) -> 'Individual':
        return Individual(
            matrix=self.matrix.copy(),
            fitness=self.fitness,
            local_search_count=self.local_search_count
        )


class StructuredPopulation:
    """
    População estruturada hierarquicamente com 13 indivíduos:
    N1:            1 (mestre)
    N2:      2     3     4    (subordinados)
    N3:    5 6 7  8 9 10  11 12 13  (trabalhadores)
    """

    def __init__(
        self,
        evaluation_function: Callable,
        local_search: VNSILS,  # updated type
        hyperparams: Dict,
        xml_path: Optional[Path] = None
    ):
        self.evaluate = evaluation_function
        self.local_search = local_search
        self.hyperparams = hyperparams
        self.logger = logging.getLogger(self.__class__.__name__)

        # Estrutura hierárquica da população
        self.hierarchy = {
            'master': 0,
            'subordinates': [1, 2, 3],
            'workers': [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]
            ]
        }

        self.individuals = []
        self._initialize_population()

    def _initialize_population(self) -> None:
        """Inicializa população hierárquica."""
        self.individuals = [
            Individual(AdaptiveMatrix(self.hyperparams)) 
            for _ in range(13)
        ]

        for ind in self.individuals[1:]:
            modified = False
            for i, aa1 in enumerate(ind.matrix.aa_order):
                for j, aa2 in enumerate(ind.matrix.aa_order[i:], i):
                    # Probabilidade menor de modificar scores altos na diagonal
                    if aa1 == aa2:
                        prob = 0.1
                    else:
                        prob = 0.3

                    if random.random() < prob:
                        current = ind.matrix.get_score(aa1, aa2)
                        adjustment = self._get_initial_adjustment(aa1, aa2)
                        new_score = current + adjustment
                        if ind.matrix._validate_score(aa1, aa2, new_score):
                            ind.matrix.update_score(aa1, aa2, new_score)
                            modified = True

            if not modified:
                self._generate_random_changes(ind.matrix)

        self.evaluate_population()

    def _get_initial_adjustment(self, aa1: str, aa2: str) -> int:
        """
        Define ajustes iniciais baseados no tipo de par de aminoácidos.
        Considera estrutura química e dados do BAliBASE4.
        """
        if aa1 == aa2:  # Diagonal
            return random.choice([-2, -1])  # Mais conservador
        elif any(aa1 in group and aa2 in group
                 for group in self.local_search.matrix.similar_groups):
            return random.choice([-2, -1, 1])  # Flexível para similares
        else:
            return random.choice([-2, -1])  # Conservador para diferentes

    def _generate_random_changes(self, matrix: AdaptiveMatrix, num_changes: int = 5) -> None:
        """Gera mudanças aleatórias respeitando restrições biológicas."""
        for _ in range(num_changes):
            aa1 = random.choice(matrix.aa_order)
            aa2 = random.choice(matrix.aa_order)
            current = matrix.get_score(aa1, aa2)

            # Ajustes mais conservadores
            if aa1 == aa2:
                adjustment = random.choice([-1, 1])
            else:
                adjustment = random.choice([-2, -1, 1])

            new_score = current + adjustment
            if matrix._validate_score(aa1, aa2, new_score):
                matrix.update_score(aa1, aa2, new_score)

    def evaluate_population(self) -> None:
        """Avalia população e ordena mantendo estrutura."""
        for ind in self.individuals:
            if ind.fitness == float('-inf'):
                ind.fitness = self.evaluate(ind.matrix)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def hierarchical_crossover(self) -> None:
        """
        Realiza crossover hierárquico completo toda geração:
        1. Trabalhadores com seus supervisores diretos (N3 com N2)
        2. Subordinados com o mestre (N2 com N1)
        """
        new_individuals = []
        for sub_idx, worker_group in zip(
            self.hierarchy['subordinates'],
            self.hierarchy['workers']
        ):
            parent = self.individuals[sub_idx]
            for worker_idx in worker_group:
                child = self._informed_crossover(parent, self.individuals[worker_idx])
                if child.fitness > self.individuals[worker_idx].fitness:
                    new_individuals.append((worker_idx, child))
        master = self.individuals[self.hierarchy['master']]
        for sub_idx in self.hierarchy['subordinates']:
            child = self._informed_crossover(master, self.individuals[sub_idx])
            if child.fitness > self.individuals[sub_idx].fitness:
                new_individuals.append((sub_idx, child))
        for idx, child in new_individuals:
            self.individuals[idx] = child
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def _informed_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover informado."""
        child = Individual(AdaptiveMatrix(self.hyperparams))
        for i, aa1 in enumerate(child.matrix.aa_order):
            for j, aa2 in enumerate(child.matrix.aa_order[i:], i):
                if random.random() < 0.5:
                    score = parent1.matrix.get_score(aa1, aa2)
                else:
                    score = parent2.matrix.get_score(aa1, aa2)
                if child.matrix._validate_score(aa1, aa2, score):
                    child.matrix.update_score(aa1, aa2, score)
        child.fitness = self.evaluate(child.matrix)
        return child


class MemeticAlgorithm:
    """Algoritmo Memético com população estruturada e VNS-ILS."""
    
    def __init__(
        self,
        evaluation_function: Callable,
        xml_path: Path,
        hyperparams: Dict
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hyperparams = hyperparams
        self.evaluation_function = evaluation_function

        # Inicializa busca local VNS com análise do XML
        self.local_search = VNSILS(
            matrix=AdaptiveMatrix(hyperparams),
            hyperparams=hyperparams
        )
        self.local_search.analyze_alignment(xml_path)

        # População estruturada reduzida
        self.population = StructuredPopulation(
            evaluation_function=evaluation_function,
            local_search=self.local_search,
            hyperparams=hyperparams,
            xml_path=xml_path
        )

        self.best_global_matrix = None
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
    ) -> Tuple[AdaptiveMatrix, float]:
        """Execução do algoritmo memético."""
        self.initial_score = self.population.individuals[0].fitness
        self.best_global_score = self.initial_score
        self.best_global_matrix = self.population.individuals[0].matrix.copy()
        stagnation_counter = 0
        generation = 0

        while (generation < generations and 
               stagnation_counter < max_no_improve and
               time.time() - self.start_time < self.hyperparams['EXECUTION']['MAX_TIME']):

            if generation % local_search_frequency == 0:
                for ind in self.population.individuals:
                    if ind.local_search_count < 3:
                        self.local_search.matrix = ind.matrix
                        new_score = self.local_search.vns_search(
                            evaluation_func=evaluation_function,
                            max_iterations=local_search_iterations,
                            max_no_improve=max_no_improve
                        )
                        if new_score > ind.fitness:
                            ind.fitness = new_score
                            ind.matrix = self.local_search.best_matrix.copy()
                            ind.local_search_count += 1
                            if new_score > self.best_global_score:
                                self.best_global_score = new_score
                                self.best_global_matrix = self.local_search.best_matrix.copy()
                                self.logger.info(f"New best score: {self.best_global_score:.4f}")
                                stagnation_counter = 0
                                continue

            self.population.hierarchical_crossover()

            current_best = self.population.individuals[0]
            if current_best.fitness > self.best_global_score:
                self.best_global_score = current_best.fitness
                self.best_global_matrix = current_best.matrix.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if generation % 5 == 0:
                elapsed = time.time() - self.start_time
                self.logger.info(
                    f"Generation {generation}: Best={self.best_global_score:.4f}, "
                    f"Time={elapsed:.1f}s, Stagnation={stagnation_counter}"
                )

            generation += 1

        total_time = time.time() - self.start_time
        self.logger.info(
            f"Optimization completed in {total_time:.1f}s. "
            f"Initial: {self.initial_score:.4f}, Final: {self.best_global_score:.4f}"
        )

        return self.best_global_matrix.copy(), self.best_global_score
