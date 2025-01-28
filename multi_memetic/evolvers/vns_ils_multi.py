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
        self.conservation_level = None  # Adicionar este atributo

        # Define estruturas de vizinhança por nível
        self.neighborhoods = {
            'HIGH': [
                VNSStructure("HIGH_CORE", 30.0, (-2, 2), 0.8),
                VNSStructure("HIGH_EXTENDED", 25.0, (-2, 2), 0.6)
            ],
            'MEDIUM': [
                VNSStructure("MEDIUM_STRICT", 22.0, (-3, 3), 0.7),
                VNSStructure("MEDIUM_FLEX", 20.0, (-3, 3), 0.5)
            ],
            'LOW': [
                VNSStructure("LOW_STRICT", 15.0, (-4, 4), 0.6),
                VNSStructure("LOW_FLEX", 10.0, (-4, 4), 0.4)
            ]
        }

        self.best_scores = {level: float('-inf') for level in ['HIGH', 'MEDIUM', 'LOW']}
        self.best_matrices = {}

    def _get_neighborhood_for_level(self, level: str) -> List[VNSStructure]:
        """Retorna vizinhanças apropriadas para o nível"""
        if level not in self.neighborhoods:
            self.logger.error(f"No neighborhoods defined for level {level}")
            return []
        return self.neighborhoods[level]

    def vns_search(
        self,
        evaluation_func: Callable,
        max_iterations: int,
        max_no_improve: int,
        conservation_level: str
    ) -> float:
        self.conservation_level = conservation_level
        neighborhoods = self._get_neighborhood_for_level(conservation_level)
        if not neighborhoods:
            raise ValueError(f"Invalid conservation level: {conservation_level}")

        current_manager = self.manager.copy()
        current_score = evaluation_func(current_manager)
        
        self.best_score = current_score
        self.best_manager = current_manager.copy()
        
        k = 0  # Índice da vizinhança atual
        iterations_no_improve = 0

        while iterations_no_improve < max_no_improve and iterations_no_improve < max_iterations:
            neighborhood = neighborhoods[k % len(neighborhoods)]
            
            # Shake
            neighbor = self._shake(current_manager, neighborhood)
            
            # Busca Local
            improved = self._quick_local_search(
                neighbor,
                evaluation_func,
                neighborhood
            )
            improved_score = evaluation_func(improved)
            
            # Move ou próxima vizinhança
            if improved_score > current_score:
                current_manager = improved
                current_score = improved_score
                k = 0

                if improved_score > self.best_score:
                    self.best_score = improved_score
                    self.best_manager = improved.copy()
                    iterations_no_improve = 0
                    continue
            else:
                k += 1
                iterations_no_improve += 1

        return self.best_score

    def _shake(
        self, 
        matrix_manager: MatrixManager,
        neighborhood: VNSStructure,
        blocks: Optional[List[Dict]] = None
    ):
        """Shake adaptado."""
        perturbed = matrix_manager.copy()
        matrix = perturbed.get_matrix(self.conservation_level)  # Pega a matriz do nível atual
        if not matrix:
            return perturbed

        changes = int(self.hyperparams['VNS']['PERTURBATION_SIZE'] * 
                     neighborhood.probability)

        for _ in range(changes):
            # Usa aa_order da matriz específica
            aa1 = random.choice(matrix.aa_order)
            aa2 = random.choice(matrix.aa_order)
            current = matrix.get_score(aa1, aa2)
            adjustment = neighborhood.get_adjustment()
            new_score = current + adjustment

            if matrix._validate_score(aa1, aa2, new_score):
                matrix.update_score(aa1, aa2, new_score)

        return perturbed

    def _quick_local_search(
        self,
        matrix_manager: MatrixManager,
        evaluation_func: Callable,
        neighborhood: VNSStructure,
        blocks: Optional[List[Dict]] = None
    ):
        """Busca local simplificada."""
        improved = matrix_manager.copy()
        improved_score = evaluation_func(improved)

        # Pega a matriz específica do nível
        matrix = improved.get_matrix(self.conservation_level)
        if not matrix:
            return improved

        for _ in range(5):  # Limite reduzido para teste
            aa1 = random.choice(matrix.aa_order)
            aa2 = random.choice(matrix.aa_order)

            current = matrix.get_score(aa1, aa2)
            adjustment = neighborhood.get_adjustment()
            new_score = current + adjustment

            if matrix._validate_score(aa1, aa2, new_score):
                candidate = improved.copy()
                candidate_matrix = candidate.get_matrix(self.conservation_level)
                candidate_matrix.update_score(aa1, aa2, new_score)
                candidate_score = evaluation_func(candidate)

                if candidate_score > improved_score:
                    improved = candidate
                    improved_score = candidate_score

        return improved

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