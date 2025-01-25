import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional, Callable
import random
from dataclasses import dataclass
from datetime import datetime

from .matrix import AdaptiveMatrix
from .local_search import EnhancedLocalSearch

@dataclass
class Individual:
    matrix: AdaptiveMatrix
    fitness: float = float('-inf')
    local_search_count: int = 0
    
    def copy(self) -> 'Individual':
        new_ind = Individual(
            matrix=self.matrix.copy(),
            fitness=self.fitness,
            local_search_count=self.local_search_count
        )
        return new_ind


class StructuredPopulation:
    def __init__(
        self,
        evaluation_function: Callable,
        local_search: EnhancedLocalSearch,
        hyperparams: Dict
    ):
        self.evaluate = evaluation_function
        self.local_search = local_search
        self.hyperparams = hyperparams
        
        self.hierarchy = {
            'master': 0,
            'subordinates': [1, 2, 3],
            'workers': [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
        }
        
        self.individuals = []
        self._initialize_population()
        
    def _initialize_population(self):
        self.individuals = [Individual(AdaptiveMatrix(self.hyperparams)) for _ in range(13)]
        
        for ind in self.individuals[1:]:
            modified = False
            for i, aa1 in enumerate(ind.matrix.aa_order):
                for j, aa2 in enumerate(ind.matrix.aa_order[i:], i):
                    if random.random() < 0.3:
                        current = ind.matrix.get_score(aa1, aa2)
                        if aa1 == aa2:
                            adjustment = random.choice([-2, -1])
                        else:
                            adjustment = random.choice([-2, -1, 1])
                        new_score = current + adjustment
                        if ind.matrix._validate_score(aa1, aa2, new_score):
                            ind.matrix.update_score(aa1, aa2, new_score)
                            modified = True
            
            if not modified:
                self._generate_random_changes(ind.matrix)
                
        self.evaluate_population()
        
    def _generate_random_changes(self, matrix: AdaptiveMatrix, num_changes: int = 5):
        for _ in range(num_changes):
            aa1 = random.choice(matrix.aa_order)
            aa2 = random.choice(matrix.aa_order)
            current = matrix.get_score(aa1, aa2)
            adjustment = random.choice([-2, -1, 1])
            new_score = current + adjustment
            if matrix._validate_score(aa1, aa2, new_score):
                matrix.update_score(aa1, aa2, new_score)
                
    def evaluate_population(self):
        for ind in self.individuals:
            if ind.fitness == float('-inf'):
                ind.fitness = self.evaluate(ind.matrix)
        
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
    def hierarchical_crossover(self):
        new_individuals = []
        
        for sub_idx, worker_group in zip(
            self.hierarchy['subordinates'],
            self.hierarchy['workers']
        ):
            parent = self.individuals[sub_idx]
            for worker_idx in worker_group:
                child = self._crossover(parent, self.individuals[worker_idx])
                mutant = self._mutate(child)
                
                if mutant.fitness > child.fitness:
                    child = mutant
                    
                if child.fitness > self.individuals[worker_idx].fitness:
                    new_individuals.append((worker_idx, child))
                    
        master = self.individuals[self.hierarchy['master']]
        for sub_idx in self.hierarchy['subordinates']:
            child = self._crossover(master, self.individuals[sub_idx])
            mutant = self._mutate(child)
            
            if mutant.fitness > child.fitness:
                child = mutant
                
            if child.fitness > self.individuals[sub_idx].fitness:
                new_individuals.append((sub_idx, child))
                
        for idx, child in new_individuals:
            self.individuals[idx] = child
            
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child = Individual(AdaptiveMatrix(self.hyperparams))
        
        better_parent = parent1 if parent1.fitness > parent2.fitness else parent2
        for i, aa in enumerate(child.matrix.aa_order):
            diag_score = better_parent.matrix.get_score(aa, aa)
            child.matrix.update_score(aa, aa, diag_score)
            
        for i, aa1 in enumerate(child.matrix.aa_order):
            for j, aa2 in enumerate(child.matrix.aa_order[i+1:], i+1):
                if random.random() < 0.5:
                    score = parent1.matrix.get_score(aa1, aa2)
                else:
                    score = parent2.matrix.get_score(aa1, aa2)
                    
                if child.matrix._validate_score(aa1, aa2, score):
                    child.matrix.update_score(aa1, aa2, score)
                    
        child.fitness = self.evaluate(child.matrix)
        return child
        
    def _mutate(self, individual: Individual) -> Individual:
        mutant = individual.copy()
        mutation_rate = self.hyperparams['MEMETIC']['MUTATION_RATE']
        
        for i, aa1 in enumerate(mutant.matrix.aa_order):
            for j, aa2 in enumerate(mutant.matrix.aa_order[i:], i):
                if random.random() < mutation_rate:
                    current = mutant.matrix.get_score(aa1, aa2)
                    adjustment = random.choice([-2, -1, 1])
                    new_score = current + adjustment
                    
                    if mutant.matrix._validate_score(aa1, aa2, new_score):
                        mutant.matrix.update_score(aa1, aa2, new_score)
                        
        mutant.fitness = self.evaluate(mutant.matrix)
        return mutant


class MemeticAlgorithm:
    def __init__(
        self,
        evaluation_function: Callable,
        xml_path: Path,
        hyperparams: Dict
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hyperparams = hyperparams
        self.evaluation_function = evaluation_function
        
        self.local_search = EnhancedLocalSearch(
            matrix=AdaptiveMatrix(hyperparams),
            hyperparams=hyperparams
        )
        
        if xml_path:
            self.local_search.analyze_alignment(xml_path)
            
        self.population = StructuredPopulation(
            evaluation_function=evaluation_function,
            local_search=self.local_search,
            hyperparams=hyperparams
        )
        
        self.best_global_matrix = None
        self.best_global_score = float('-inf')
        self.initial_score = None
        
        self.initialize_population()
        
    def initialize_population(self):
        self.population.evaluate_population()
        best_individual = self.population.individuals[0]
        
        self.best_global_matrix = best_individual.matrix.copy()
        self.best_global_score = best_individual.fitness
        self.initial_score = best_individual.fitness
        self.logger.info(f"Initial best fitness: {self.initial_score:.4f}")
        
    def run(
        self,
        generations: int,
        local_search_frequency: int,
        local_search_iterations: int,
        max_no_improve: int,
        evaluation_function: Callable
    ) -> Tuple[AdaptiveMatrix, float]:
        self.logger.info("Starting memetic optimization")
        stagnation_counter = 0
        
        for generation in range(generations):
            self.population.evaluate_population()
            
            if generation % local_search_frequency == 0:
                self._apply_local_search(
                    local_search_iterations,
                    max_no_improve,
                    evaluation_function
                )
            
            self.population.hierarchical_crossover()
            
            current_best = self.population.individuals[0]
            if current_best.fitness > self.best_global_score:
                self.best_global_score = current_best.fitness
                self.best_global_matrix = current_best.matrix.copy()
                stagnation_counter = 0
                self.logger.info(f"New best score: {self.best_global_score:.4f}")
            else:
                stagnation_counter += 1
                
            if stagnation_counter >= max_no_improve:
                self.logger.info(f"Stopping early due to stagnation after {generation} generations")
                break
                
            if generation % 5 == 0:
                self.logger.info(
                    f"Generation {generation}: "
                    f"Best={self.best_global_score:.4f}, "
                    f"Master={self.population.individuals[0].fitness:.4f}, "
                    f"Stagnation={stagnation_counter}"
                )
                
        if self.best_global_matrix is not None:
            final_score = evaluation_function(self.best_global_matrix)
            return self.best_global_matrix.copy(), final_score
            
        return None, float('-inf')
        
    def _apply_local_search(
        self,
        local_search_iterations: int,
        max_no_improve: int,
        evaluation_function: Callable
    ) -> None:
        for idx in [self.population.hierarchy['master']] + self.population.hierarchy['subordinates']:
            individual = self.population.individuals[idx]
            if individual.local_search_count < 3:
                self.local_search.matrix = individual.matrix
                new_score = self.local_search.vns_search(
                    evaluation_func=evaluation_function,
                    max_iterations=local_search_iterations,
                    max_no_improve=max_no_improve
                )
                
                if new_score > individual.fitness:
                    individual.fitness = new_score
                    individual.matrix = self.local_search.best_matrix.copy()
                    
                    if new_score > self.best_global_score:
                        self.best_global_score = new_score
                        self.best_global_matrix = self.local_search.best_matrix.copy()
                        
                individual.local_search_count += 1
                
        self.population.evaluate_population()