# multi_memetic/evolvers/vns_ils_multi.py

import logging
import random
from typing import Dict, List, Set, Tuple, Optional, Callable
import numpy as np

from multi_memetic.adaptive_matrices.matrix_manager import MatrixManager
from multi_memetic.utils.xml_parser import ScoreAccessLayer, ConservationLevel

class VNSStructure:
    """Define estrutura e operadores de vizinhança para cada nível"""

    def __init__(
        self,
        name: str,
        block_score_min: float,
        adjustment_range: Tuple[int, int],
        probability: float
    ):
        self.name = name
        self.block_score_min = block_score_min
        self.adjustment_range = adjustment_range
        self.probability = probability

    def get_adjustment(self) -> int:
        """Retorna ajuste aleatório dentro do range permitido"""
        return random.randint(*self.adjustment_range)

class VNSILS:
    """VNS-ILS adaptado para otimização multinível"""

    def __init__(self, manager: MatrixManager, hyperparams: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.manager = manager
        self.hyperparams = hyperparams

        # Define estruturas de vizinhança por nível
        self.neighborhoods = {
            ConservationLevel.HIGH: [
                VNSStructure("HIGH_CORE", 30.0, (-2, 2), 0.8),
                VNSStructure("HIGH_EXTENDED", 25.0, (-2, 2), 0.6)
            ],
            ConservationLevel.MEDIUM: [
                VNSStructure("MEDIUM_STRICT", 22.0, (-3, 3), 0.7),
                VNSStructure("MEDIUM_FLEX", 20.0, (-3, 3), 0.5)
            ],
            ConservationLevel.LOW: [
                VNSStructure("LOW_STRICT", 15.0, (-4, 4), 0.6),
                VNSStructure("LOW_FLEX", 10.0, (-4, 4), 0.4)
            ]
        }

        self.best_scores = {level: float('-inf') for level in ConservationLevel.__dict__
                            if not level.startswith('_')}
        self.best_matrices = {}

    def vns_search(
        self,
        evaluation_func: Callable,
        max_iterations: int,
        max_no_improve: int,
        conservation_level: str,
        blocks: List[Dict]
    ) -> float:
        """VNS adaptado para nível específico de conservação"""

        matrix = self.manager.get_matrix(conservation_level)
        if not matrix:
            return float('-inf')

        current_matrix = matrix.copy()
        current_score = evaluation_func(current_matrix)

        self.best_scores[conservation_level] = current_score
        self.best_matrices[conservation_level] = current_matrix.copy()

        neighborhoods = self.neighborhoods[conservation_level]
        k = 0  # índice da vizinhança atual
        iterations_no_improve = 0

        while (iterations_no_improve < max_no_improve and
               k < len(neighborhoods)):

            neighborhood = neighborhoods[k]

            # Shake
            neighbor = self._shake(current_matrix, neighborhood, blocks)
            neighbor_score = evaluation_func(neighbor)

            # Local Search
            improved = self._quick_local_search(
                neighbor,
                evaluation_func,
                neighborhood,
                blocks
            )
            improved_score = evaluation_func(improved)

            # Move ou próxima vizinhança
            if improved_score > current_score + self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                current_matrix = improved
                current_score = improved_score
                k = 0  # reinicia vizinhanças

                if improved_score > self.best_scores[conservation_level]:
                    self.best_scores[conservation_level] = improved_score
                    self.best_matrices[conservation_level] = improved.copy()
                    iterations_no_improve = 0
                    continue
            else:
                k += 1
                iterations_no_improve += 1

        return self.best_scores[conservation_level]

    def _quick_local_search(
        self,
        matrix,
        evaluation_func: Callable,
        neighborhood: VNSStructure,
        blocks: List[Dict]
    ):
        """Busca local rápida focada em blocos relevantes"""
        improved = matrix.copy()
        improved_score = evaluation_func(improved)

        relevant_blocks = [b for b in blocks
                           if b['score'] >= neighborhood.block_score_min]

        if not relevant_blocks:
            return improved

        for block in relevant_blocks:
            for _ in range(5):  # Limite de tentativas por bloco
                aa1 = random.choice(matrix.aa_order)
                aa2 = random.choice(matrix.aa_order)

                current = improved.get_score(aa1, aa2)
                adjustment = neighborhood.get_adjustment()
                new_score = current + adjustment

                if improved._validate_score(aa1, aa2, new_score):
                    candidate = improved.copy()
                    candidate.update_score(aa1, aa2, new_score)
                    candidate_score = evaluation_func(candidate)

                    if candidate_score > improved_score + \
                       self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                        improved = candidate
                        improved_score = candidate_score

        return improved

    def _shake(self, matrix, neighborhood: VNSStructure, blocks: List[Dict]):
        """Shake adaptado ao nível de conservação"""
        relevant_blocks = [b for b in blocks
                           if b['score'] >= neighborhood.block_score_min]

        if not relevant_blocks:
            return matrix.copy()

        perturbed = matrix.copy()
        changes = int(self.hyperparams['VNS']['PERTURBATION_SIZE'] *
                     neighborhood.probability)

        for _ in range(changes):
            block = random.choice(relevant_blocks)
            aa1 = random.choice(matrix.aa_order)
            aa2 = random.choice(matrix.aa_order)

            current = matrix.get_score(aa1, aa2)
            adjustment = neighborhood.get_adjustment()
            new_score = current + adjustment

            if matrix._validate_score(aa1, aa2, new_score):
                perturbed.update_score(aa1, aa2, new_score)

        return perturbed

    def _local_search(
        self,
        matrix: MatrixManager,
        evaluation_func: Callable,
        neighborhood: VNSStructure,
        blocks: List[Dict]
    ) -> MatrixManager:
        """Busca local completa para blocos de um nível"""
        improved = matrix.copy()
        improved_score = evaluation_func(improved)

        relevant_blocks = [b for b in blocks if b['score'] >= neighborhood.block_score_min]
        
        if not relevant_blocks:
            return improved

        for block in relevant_blocks:
            block_improved = True
            while block_improved:
                block_improved = False

                for i, aa1 in enumerate(improved.aa_order):
                    for aa2 in improved.aa_order[i:]:
                        current = improved.get_score(aa1, aa2)
                        for adj in range(*neighborhood.adjustment_range):
                            candidate = improved.copy()
                            new_score = current + adj

                            if candidate._validate_score(aa1, aa2, new_score):
                                candidate.update_score(aa1, aa2, new_score)
                                candidate_score = evaluation_func(candidate)

                                if candidate_score > improved_score + \
                                   self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                                    improved = candidate
                                    improved_score = candidate_score
                                    block_improved = True
                                    break

                        if block_improved:
                            break
                    if block_improved:
                        break

        return improved