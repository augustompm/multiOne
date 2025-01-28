# multi_memetic/evolvers/population_multi.py

import numpy as np
import logging
from typing import Dict, List, Optional, Callable
import random
from dataclasses import dataclass
from pathlib import Path
import time

from multi_memetic.adaptive_matrices.matrix_manager import MatrixManager
from multi_memetic.utils.xml_parser import ScoreAccessLayer, ConservationLevel

@dataclass
class IndividualMulti:
    """
    Indivíduo do algoritmo memético que mantém três matrizes adaptativas
    através do MatrixManager.
    """
    matrix_manager: MatrixManager
    fitness: float = float('-inf')
    local_search_count: int = 0
    
    def copy(self) -> 'IndividualMulti':
        """Cria cópia profunda do indivíduo"""
        return IndividualMulti(
            matrix_manager=self.matrix_manager.copy(),
            fitness=self.fitness,
            local_search_count=self.local_search_count
        )

class StructuredPopulationMulti:
    """
    População estruturada hierarquicamente com 13 indivíduos:
    N1:            1 (mestre)
    N2:      2     3     4    (subordinados)
    N3:   5 6 7  8 9 10  11 12 13  (trabalhadores)
    
    Cada indivíduo contém três matrizes adaptativas.
    """
    def __init__(
        self,
        evaluation_function: Callable,
        hyperparams: Dict,
        reference_analysis: Optional[Dict] = None
    ):
        self.evaluate = evaluation_function
        self.hyperparams = hyperparams
        self.reference_analysis = reference_analysis
        self.logger = logging.getLogger(self.__class__.__name__)

        # Estrutura hierárquica mantida do original
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
        """Inicializa população hierárquica com indivíduos multi-matriz"""
        self.individuals = [
            IndividualMulti(MatrixManager(self.hyperparams)) 
            for _ in range(13)
        ]

        # Modifica matrizes dos indivíduos não-master
        for ind in self.individuals[1:]:
            modified = False
            for level in [ConservationLevel.HIGH, ConservationLevel.MEDIUM, 
                         ConservationLevel.LOW]:
                matrix = ind.matrix_manager.get_matrix(level)
                
                # Aplica modificações aleatórias para cada matriz
                for i, aa1 in enumerate(matrix.aa_order):
                    for j, aa2 in enumerate(matrix.aa_order[i:], i):
                        # Probabilidade menor de modificar scores altos na diagonal
                        prob = 0.1 if aa1 == aa2 else 0.3
                        
                        if random.random() < prob:
                            current = matrix.get_score(aa1, aa2)
                            adjustment = self._get_initial_adjustment(aa1, aa2, level)
                            new_score = current + adjustment
                            if matrix._validate_score(aa1, aa2, new_score):
                                matrix.update_score(aa1, aa2, new_score)
                                modified = True

            if not modified:
                self._generate_random_changes(ind.matrix_manager)

        self.evaluate_population()

    def _get_initial_adjustment(self, aa1: str, aa2: str, level: str) -> int:
        """Define ajustes iniciais baseado no nível de conservação"""
        if level == 'HIGH':
            return random.choice([-1, 1])  # Mais conservador
        elif level == 'MEDIUM':  
            return random.choice([-2, -1, 1, 2])  # Flexibilidade média
        else:  # LOW
            return random.choice([-3, -2, -1, 1, 2, 3])  # Mais flexível

    def _generate_random_changes(self, manager: MatrixManager, changes_per_matrix: int = 5) -> None:
        """Gera mudanças aleatórias para cada matriz"""
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            matrix = manager.get_matrix(level)
            if matrix:
                for _ in range(changes_per_matrix):
                    aa1 = random.choice(matrix.aa_order)
                    aa2 = random.choice(matrix.aa_order)
                    current = matrix.get_score(aa1, aa2)
                    adjustment = self._get_initial_adjustment(aa1, aa2, level)
                    new_score = current + adjustment
                    if matrix._validate_score(aa1, aa2, new_score):
                        matrix.update_score(aa1, aa2, new_score)

    def evaluate_population(self) -> None:
        """Avalia população e ordena mantendo estrutura"""
        for ind in self.individuals:
            if ind.fitness == float('-inf'):
                ind.fitness = self.evaluate(ind.matrix_manager)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def hierarchical_crossover(self) -> None:
        """
        Realiza crossover hierárquico:
        1. Trabalhadores com supervisores (N3 com N2)
        2. Subordinados com mestre (N2 com N1)
        """
        new_individuals = []
        
        # Crossover N3-N2
        for sub_idx, worker_group in zip(
            self.hierarchy['subordinates'],
            self.hierarchy['workers']
        ):
            parent = self.individuals[sub_idx]
            for worker_idx in worker_group:
                child = self._informed_crossover(parent, self.individuals[worker_idx])
                if child.fitness > self.individuals[worker_idx].fitness:
                    new_individuals.append((worker_idx, child))
        
        # Crossover N2-N1
        master = self.individuals[self.hierarchy['master']]
        for sub_idx in self.hierarchy['subordinates']:
            child = self._informed_crossover(master, self.individuals[sub_idx])
            if child.fitness > self.individuals[sub_idx].fitness:
                new_individuals.append((sub_idx, child))
        
        # Atualiza população
        for idx, child in new_individuals:
            self.individuals[idx] = child
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def _informed_crossover(self, parent1: IndividualMulti, 
                          parent2: IndividualMulti) -> IndividualMulti:
        """Crossover informado mantendo estrutura de três matrizes"""
        child = IndividualMulti(MatrixManager(self.hyperparams))
        
        # Crossover para cada matriz
        for level in [ConservationLevel.HIGH, ConservationLevel.MEDIUM, 
                     ConservationLevel.LOW]:
            matrix_child = child.matrix_manager.get_matrix(level)
            matrix_p1 = parent1.matrix_manager.get_matrix(level)
            matrix_p2 = parent2.matrix_manager.get_matrix(level)
            
            for i, aa1 in enumerate(matrix_child.aa_order):
                for j, aa2 in enumerate(matrix_child.aa_order[i:], i):
                    if random.random() < 0.5:
                        score = matrix_p1.get_score(aa1, aa2)
                    else:
                        score = matrix_p2.get_score(aa1, aa2)
                        
                    if matrix_child._validate_score(aa1, aa2, score):
                        matrix_child.update_score(aa1, aa2, score)
        
        child.fitness = self.evaluate(child.matrix_manager)
        return child