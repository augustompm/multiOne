# memetic/memetic.py

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from copy import deepcopy
import random

from .matrix import AdaptiveMatrix
from .local_search import LocalSearch

class Individual:
    """
    Represents a single solution in our population - an adaptive substitution matrix
    with its associated fitness score and optimization history.
    """
    def __init__(self, matrix: Optional[AdaptiveMatrix] = None):
        # Initialize with either a provided matrix or create a new one
        self.matrix = matrix if matrix else AdaptiveMatrix()
        self.fitness = float('-inf')
        self.improvement_count = 0
        self.local_search_applications = 0
        
    def copy(self) -> 'Individual':
        """Creates a deep copy of the individual."""
        new_ind = Individual()
        new_ind.matrix = deepcopy(self.matrix)
        new_ind.fitness = self.fitness
        new_ind.improvement_count = self.improvement_count
        new_ind.local_search_applications = self.local_search_applications
        return new_ind

class ElitePool:
    """
    Manages the elite individuals in our population, maintaining diversity
    while preserving the best solutions found.
    """
    def __init__(self, size: int, diversity_threshold: float = 0.1):
        self.size = size
        self.diversity_threshold = diversity_threshold
        self.individuals: List[Individual] = []
        
    def add(self, individual: Individual) -> bool:
        """
        Attempts to add an individual to the elite pool while maintaining diversity.
        Returns True if individual was added.
        """
        # If pool isn't full, add directly
        if len(self.individuals) < self.size:
            self.individuals.append(individual.copy())
            self._sort_pool()
            return True
            
        # Check if new individual is better than worst elite
        if individual.fitness <= self.individuals[-1].fitness:
            return False
            
        # Check diversity against existing elite members
        for elite in self.individuals:
            if self._matrix_similarity(individual.matrix, elite.matrix) > (1 - self.diversity_threshold):
                # Too similar to existing elite - replace if better
                if individual.fitness > elite.fitness:
                    elite.matrix = individual.matrix
                    elite.fitness = individual.fitness
                    self._sort_pool()
                return False
                
        # Add new individual, remove worst
        self.individuals[-1] = individual.copy()
        self._sort_pool()
        return True
        
    def _sort_pool(self):
        """Sorts elite pool by fitness in descending order."""
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
    def _matrix_similarity(self, matrix1: AdaptiveMatrix, matrix2: AdaptiveMatrix) -> float:
        """
        Calculates similarity between two matrices as proportion of entries
        within a small difference threshold.
        """
        diff = np.abs(matrix1.matrix - matrix2.matrix)
        return np.mean(diff <= 1)  # Consider entries within ±1 as similar
        
    def get_best(self) -> Individual:
        """Returns the best individual in the pool."""
        return self.individuals[0].copy() if self.individuals else None
        
    def get_random_elite(self) -> Individual:
        """Returns a random elite individual."""
        return random.choice(self.individuals).copy() if self.individuals else None

class Population:
    """
    Manages a population of adaptive matrices, handling their evolution
    and interaction with local search.
    """
    def __init__(self, 
                 size: int,
                 elite_size: int,
                 evaluation_function,
                 local_search: LocalSearch):
        self.size = size
        self.individuals: List[Individual] = []
        self.elite_pool = ElitePool(elite_size)
        self.evaluate = evaluation_function
        self.local_search = local_search
        
        # Initialize population
        self._initialize_population()
        
    def _initialize_population(self):
        """Creates initial population with controlled diversity."""
        # Create first individual from standard PAM250
        first_ind = Individual()
        self.individuals.append(first_ind)
        
        # Create rest with small random perturbations
        while len(self.individuals) < self.size:
            new_ind = Individual()
            # Apply small random changes to matrix
            for i in range(20):
                for j in range(i+1, 20):
                    if random.random() < 0.1:  # 10% chance of modification
                        current = new_ind.matrix.get_score(
                            new_ind.matrix.aa_order[i],
                            new_ind.matrix.aa_order[j]
                        )
                        # Small random adjustment
                        adjustment = random.choice([-1, 1])
                        new_ind.matrix.update_score(
                            new_ind.matrix.aa_order[i],
                            new_ind.matrix.aa_order[j],
                            current + adjustment
                        )
            self.individuals.append(new_ind)
            
    def evaluate_population(self):
        """Evaluates all individuals in the population."""
        for ind in self.individuals:
            if ind.fitness == float('-inf'):  # Only evaluate if not already scored
                ind.fitness = self.evaluate(ind.matrix)
                
    def update_elite(self):
        """Updates elite pool with current population."""
        for ind in self.individuals:
            self.elite_pool.add(ind)
            
    def apply_local_search(self, 
                          max_iterations: int,
                          max_no_improve: int,
                          proportion: float = 0.2):
        """
        Applies local search to a proportion of the population,
        focusing on promising individuals.
        """
        # Sort population by fitness
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
        # Apply local search to top proportion
        num_to_improve = int(self.size * proportion)
        for i in range(num_to_improve):
            ind = self.individuals[i]
            
            # Skip if local search recently applied
            if ind.local_search_applications >= 2:
                continue
                
            # Apply VNS local search
            self.local_search.matrix = ind.matrix
            new_score = self.local_search.vns_search(
                self.evaluate,
                max_iterations=max_iterations,
                max_no_improve=max_no_improve
            )
            
            if new_score > ind.fitness:
                ind.fitness = new_score
                ind.improvement_count += 1
                
            ind.local_search_applications += 1
            
    def create_next_generation(self):
        """
        Creates next generation using elite solutions.
        Note: Currently only uses elite solutions as basis,
        mutation/crossover to be added later.
        """
        new_population = []
        
        # Keep elite solutions
        for elite in self.elite_pool.individuals:
            new_population.append(elite.copy())
            
        # Fill rest with variations of elite solutions
        while len(new_population) < self.size:
            # Select random elite
            template = self.elite_pool.get_random_elite()
            new_ind = template.copy()
            
            # Apply small random changes
            for i in range(20):
                for j in range(i+1, 20):
                    if random.random() < 0.05:  # 5% chance of modification
                        current = new_ind.matrix.get_score(
                            new_ind.matrix.aa_order[i],
                            new_ind.matrix.aa_order[j]
                        )
                        adjustment = random.choice([-1, 1])
                        new_ind.matrix.update_score(
                            new_ind.matrix.aa_order[i],
                            new_ind.matrix.aa_order[j],
                            current + adjustment
                        )
                        
            new_population.append(new_ind)
            
        self.individuals = new_population

class MemeticAlgorithm:
    """
    Coordinates the memetic algorithm, combining population-based search
    with local improvement.
    """
    def __init__(self,
                 population_size: int,
                 elite_size: int = None,  # Will default to 10% of population
                 evaluation_function = None,
                 xml_path: Path = None,
                 max_generations: int = 100,
                 local_search_frequency: int = 5):
        
        # Set default elite size if not provided
        if elite_size is None:
            elite_size = max(3, int(population_size * 0.1))
            
        self.population_size = population_size
        self.max_generations = max_generations
        self.local_search_frequency = local_search_frequency
        
        # Initialize local search
        self.local_search = LocalSearch(AdaptiveMatrix())
        if xml_path:
            self.local_search.analyze_alignment(xml_path)
            
        # Initialize population
        self.population = Population(
            size=population_size,
            elite_size=elite_size,
            evaluation_function=evaluation_function,
            local_search=self.local_search
        )
        
    def run(self,
            generations: int,
            local_search_frequency: int,
            local_search_iterations: int,
            max_no_improve: int) -> AdaptiveMatrix:
        """
        Runs the memetic optimization process.
        Returns the best matrix found.
        """
        for generation in range(generations):
            # Evaluate current population
            self.population.evaluate_population()
            
            # Update elite pool
            self.population.update_elite()
            
            # Apply local search periodically
            if generation % local_search_frequency == 0:
                self.population.apply_local_search(
                    max_iterations=local_search_iterations,
                    max_no_improve=max_no_improve
                )
                
            # Create next generation
            self.population.create_next_generation()
            
            # Log progress
            best_fitness = self.population.elite_pool.get_best().fitness
            logging.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.4f}")
            
        return self.population.elite_pool.get_best().matrix