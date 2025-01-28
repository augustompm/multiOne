# multi_memetic/evolvers/vns_ils_multi.py

import logging
import random
from typing import Dict, List, Set, Tuple, Optional, Callable
import numpy as np

from multi_memetic.adaptive_matrices.matrix_manager import MatrixManager
from multi_memetic.utils.xml_parser import ScoreAccessLayer, ConservationLevel

class VNSStructure:
    """Define estrutura e operadores de uma vizinhança VNS."""

    def __init__(
        self,
        name: str,
        min_score: float,
        max_adjustment: int,
        probability: float,
        description: str
    ):
        self.name = name
        self.min_score = min_score
        self.max_adjustment = max_adjustment
        self.probability = probability
        self.description = description

    def get_adjustment_range(self) -> List[int]:
        """Define range de ajustes para a vizinhança."""
        return list(range(-self.max_adjustment, self.max_adjustment + 1))

class VNSILS:
    """
    VNS-ILS adaptado para otimizar matrizes específicas para cada 
    nível de conservação.
    """

    def __init__(self, matrix_manager: MatrixManager, hyperparams: Dict,
                 xml_parser: ScoreAccessLayer):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.matrix_manager = matrix_manager
        self.hyperparams = hyperparams
        self.xml_parser = xml_parser
        self.conservation_level = None  # Adiciona o nível de conservação

        # Define vizinhanças formais baseadas em scores reais do BAliBASE
        self.neighborhoods = {
            ConservationLevel.HIGH: [
                VNSStructure(
                    "HIGH_STRICT",
                    min_score=30.0,
                    max_adjustment=2,
                    probability=0.8,
                    description="Blocos altamente conservados - ajustes restritos"
                ),
                VNSStructure(
                    "HIGH_MEDIUM",
                    min_score=25.0,
                    max_adjustment=3,
                    probability=0.6,
                    description="Blocos altamente conservados - ajustes médios"
                )
            ],
            ConservationLevel.MEDIUM: [
                VNSStructure(
                    "MEDIUM_STRICT",
                    min_score=22.0,
                    max_adjustment=3,
                    probability=0.7,
                    description="Blocos médios - ajustes restritos"
                ),
                VNSStructure(
                    "MEDIUM_FLEX",
                    min_score=20.0,
                    max_adjustment=4,
                    probability=0.5,
                    description="Blocos médios - ajustes flexíveis"
                )
            ],
            ConservationLevel.LOW: [
                VNSStructure(
                    "LOW_STRICT",
                    min_score=15.0,
                    max_adjustment=4,
                    probability=0.6,
                    description="Blocos baixos - ajustes base"
                ),
                VNSStructure(
                    "LOW_FLEX",
                    min_score=10.0,
                    max_adjustment=5,
                    probability=0.4,
                    description="Blocos baixos - ajustes flexíveis"
                )
            ]
        }

        self.best_matrix = None
        self.best_score = float('-inf')

    def vns_search(
        self,
        evaluation_func: Callable,
        max_iterations: int,
        max_no_improve: int,
        blocks: List[Dict],
        conservation_level: str = 'HIGH'  # Valor default para compatibilidade
    ) -> float:
        """VNS-ILS adaptado para trabalhar com o gerenciador ao invés da matriz individual"""

        self.conservation_level = conservation_level  # Define o nível de conservação

        # Mudança: trabalhar com o gerenciador ao invés da matriz individual
        current_manager = self.matrix_manager.copy()
        current_score = evaluation_func(current_manager)
        
        self.best_score = current_score
        self.best_matrix = current_manager.copy()
        
        k = 0  # Índice da vizinhança atual
        iterations_no_improve = 0 
        total_iterations = 0
        
        self.logger.info(
            f"Starting VNS search for {conservation_level} from score: {current_score:.4f}")
        
        neighborhoods = self.neighborhoods[conservation_level]
        
        while (iterations_no_improve < max_no_improve and 
               total_iterations < max_iterations):
            
            self.logger.debug(
                f"Iter {total_iterations}, N{k+1}, Score: {current_score:.4f}, "
                f"NoImprove: {iterations_no_improve}")
            
            # 1. Shaking
            neighborhood = neighborhoods[k]
            neighbor = self._shake(current_manager, neighborhood, blocks)
            
            # 2. Busca Local - reduzido para debug
            improved = self._quick_local_search(
                neighbor,
                evaluation_func,
                neighborhood,
                blocks
            )
            improved_score = evaluation_func(improved)
            
            # 3. Move ou Próxima Vizinhança
            if improved_score > current_score + \
               self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                current_manager = improved
                current_score = improved_score
                k = 0  # Volta para primeira vizinhança
                
                if improved_score > self.best_score:
                    self.best_score = improved_score
                    self.best_matrix = improved.copy()
                    self.logger.info(
                        f"New best score {improved_score:.4f} in N{k+1}")
                    iterations_no_improve = 0
                    continue
            else:
                k = (k + 1) % len(neighborhoods)
                iterations_no_improve += 1
                
            total_iterations += 1
                
        return self.best_score

    def _quick_local_search(
        self,
        matrix_manager: MatrixManager,
        evaluation_func: Callable,
        neighborhood: VNSStructure,
        blocks: List[Dict]
    ) -> MatrixManager:
        """Busca local simplificada por nível de conservação"""
        improved = matrix_manager.copy()
        improved_score = evaluation_func(improved)
        
        # Obtém a matriz específica para o nível de conservação
        current_matrix = improved.get_matrix(self.conservation_level)
        
        # Tenta ajustes apenas em blocos do nível correto
        for block in blocks:
            if block['score'] >= neighborhood.min_score:
                for _ in range(5):  # Limite pequeno para teste
                    aa1 = random.choice(current_matrix.aa_order)
                    aa2 = random.choice(current_matrix.aa_order)
                    
                    current = current_matrix.get_score(aa1, aa2)
                    adjustment = random.choice([-1, 1])
                    new_score = current + adjustment
                    
                    if current_matrix._validate_score(aa1, aa2, new_score):
                        candidate = improved.copy()
                        candidate_matrix = candidate.get_matrix(self.conservation_level)
                        candidate_matrix.update_score(aa1, aa2, new_score)
                        candidate_score = evaluation_func(candidate)
                        
                        if candidate_score > improved_score + \
                           self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                            improved = candidate
                            improved_score = candidate_score
                    
        return improved

    def _shake(
        self,
        matrix_manager: MatrixManager,
        neighborhood: VNSStructure,
        blocks: List[Dict]
    ) -> MatrixManager:
        """Shake sistemático para blocos de um nível específico"""
        perturbed = matrix_manager.copy()

        relevant_blocks = [b for b in blocks if b['score'] >= neighborhood.min_score]
        
        if not relevant_blocks:
            return perturbed

        changes = int(
            self.hyperparams['VNS']['PERTURBATION_SIZE'] *
            neighborhood.probability
        )
        
        # Obtém a matriz específica para o nível de conservação
        current_matrix = perturbed.get_matrix(self.conservation_level)

        for _ in range(changes):
            block = random.choice(relevant_blocks)
            aa1 = random.choice(current_matrix.aa_order)
            aa2 = random.choice(current_matrix.aa_order)

            current = current_matrix.get_score(aa1, aa2)
            adjustment = random.choice(neighborhood.get_adjustment_range())
            new_score = current + adjustment

            if current_matrix._validate_score(aa1, aa2, new_score):
                current_matrix.update_score(aa1, aa2, new_score)

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

        relevant_blocks = [b for b in blocks if b['score'] >= neighborhood.min_score]
        
        if not relevant_blocks:
            return improved

        for block in relevant_blocks:
            block_improved = True
            while block_improved:
                block_improved = False

                for i, aa1 in enumerate(improved.aa_order):
                    for aa2 in improved.aa_order[i:]:
                        current = improved.get_score(aa1, aa2)
                        for adj in neighborhood.get_adjustment_range():
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