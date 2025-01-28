import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from collections import defaultdict
import logging
import random
from typing import Dict, List, Set, Tuple, Optional, Callable
from .matrix import AdaptiveMatrix

class ScoreAccessLayer:
    """Camada de abstração para acessar scores do alinhamento."""
    def __init__(self, logger):
        self.column_scores = {}
        self.column_score_owners = {}
        self.group_score_mapping = {}
        self.logger = logger
        self.block_scores = defaultdict(list)  # Scores reais dos blocos
        self.color_mapping = defaultdict(set)  # Mapeia cores para blocos relacionados
        self.disorder_regions = set()          # Regiões DISORDER marcadas

    def load_from_xml(self, root) -> None:
        """Carrega dados preservando nomenclaturas diferentes entre XMLs."""
        self.column_scores.clear()
        self.column_score_owners.clear()
        self.group_score_mapping.clear()
        self.block_scores.clear()
        self.color_mapping.clear()
        self.disorder_regions.clear()

        # Carrega column-scores
        for colsco in root.findall(".//column-score"):
            try:
                name = colsco.find("colsco-name").text
                owner = colsco.find("colsco-owner").text if colsco.find("colsco-owner") is not None else ""
                data_text = colsco.find("colsco-data").text
                if not data_text:
                    continue

                data = [int(x) for x in data_text.split()]
                self.column_scores[name] = data

                if owner:
                    self.column_score_owners[name] = owner

                # Mapeia grupos
                if "normd_" in name or "group" in owner:
                    try:
                        group_id = int(''.join(filter(str.isdigit, name if "normd_" in name else owner)))
                        self.group_score_mapping[group_id] = name
                    except ValueError:
                        continue

            except Exception as e:
                self.logger.debug(f"Skipping score {name}: {e}")
                continue

        # Carrega blocos e seus scores
        for seq in root.findall(".//sequence"):
            for block in seq.findall(".//fitem"):
                if block.find("ftype").text == "BLOCK":
                    try:
                        score = float(block.find("fscore").text)
                        color = int(block.find("fcolor").text)
                        start = int(block.find("fstart").text)
                        stop = int(block.find("fstop").text)

                        self.block_scores[score].append({
                            'start': start,
                            'stop': stop,
                            'color': color,
                            'length': stop - start + 1
                        })
                        self.color_mapping[color].add(score)

                    except Exception:
                        continue

                elif block.find("ftype").text == "DISORDER":
                    try:
                        start = int(block.find("fstart").text)
                        stop = int(block.find("fstop").text)
                        self.disorder_regions.add((start, stop))
                    except Exception:
                        continue

    def get_block_boundaries(self, min_score: float) -> List[Tuple[int, int]]:
        """Retorna limites dos blocos com score mínimo especificado."""
        boundaries = []
        for score, blocks in self.block_scores.items():
            if score >= min_score:
                for block in blocks:
                    boundaries.append((block['start'], block['stop']))
        return sorted(boundaries)

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
    Implementação formal de VNS-ILS para otimização de matrizes.
    Usa estrutura real do BAliBASE4 para definir vizinhanças.
    """

    def __init__(self, matrix: AdaptiveMatrix, hyperparams: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.matrix = matrix
        self.hyperparams = hyperparams

        # Define vizinhanças formais baseadas em scores reais do BAliBASE
        self.neighborhoods = [
            VNSStructure(
                "HIGH_CONSERVATION",
                min_score=25.0,
                max_adjustment=2,
                probability=0.8,
                description="Blocos altamente conservados"
            ),
            VNSStructure(
                "MEDIUM_CONSERVATION",
                min_score=20.0,
                max_adjustment=3,
                probability=0.6,
                description="Blocos moderadamente conservados"
            ),
            VNSStructure(
                "LOW_CONSERVATION",
                min_score=15.0,
                max_adjustment=4,
                probability=0.4,
                description="Blocos pouco conservados"
            )
        ]

        # Acesso resiliente aos scores
        self.scores = ScoreAccessLayer(self.logger)

        self.best_matrix = None
        self.best_score = float('-inf')

    def analyze_alignment(self, xml_path: Path) -> None:
        """Analisa alinhamento de referência."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            self.scores.load_from_xml(root)
            self.logger.info(
                f"Analyzed alignment: {len(self.scores.block_scores)} blocks, "
                f"{len(self.scores.disorder_regions)} disorder regions"
            )
        except Exception as e:
            self.logger.error(f"Error analyzing alignment: {e}")
            raise

    def vns_search(
        self,
        evaluation_func: Callable,
        max_iterations: int,
        max_no_improve: int
    ) -> float:
        """VNS-ILS com logs de debug."""
        current_matrix = self.matrix.copy()
        current_score = evaluation_func(current_matrix)
        
        self.best_score = current_score
        self.best_matrix = current_matrix.copy()
        
        k = 0  # Índice da vizinhança atual
        iterations_no_improve = 0 
        total_iterations = 0
        
        self.logger.info(f"Starting VNS search from score: {current_score:.4f}")
        
        while iterations_no_improve < max_no_improve and total_iterations < max_iterations:
            # Debug
            self.logger.debug(
                f"Iter {total_iterations}, N{k+1}, "
                f"Score: {current_score:.4f}, NoImprove: {iterations_no_improve}"
            )
            
            # 1. Shaking
            neighborhood = self.neighborhoods[k]
            neighbor = self._shake(current_matrix, neighborhood)
            
            # 2. Busca Local - reduzido para debug
            improved = self._quick_local_search(
                neighbor,
                evaluation_func,
                neighborhood
            )
            improved_score = evaluation_func(improved)
            
            # 3. Move ou Próxima Vizinhança
            if improved_score > current_score + self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                current_matrix = improved
                current_score = improved_score
                k = 0  # Volta para primeira vizinhança
                
                if improved_score > self.best_score:
                    self.best_score = improved_score
                    self.best_matrix = improved.copy()
                    self.logger.info(f"New best score {improved_score:.4f} in N{k+1}")
                    iterations_no_improve = 0
                    continue
            else:
                k = (k + 1) % len(self.neighborhoods)
                iterations_no_improve += 1
                
            total_iterations += 1
                
        return self.best_score
    
    def _quick_local_search(
        self,
        matrix: AdaptiveMatrix,
        evaluation_func: Callable,
        neighborhood: VNSStructure  
    ) -> AdaptiveMatrix:
        """Versão simplificada da busca local para debug."""
        improved = matrix.copy()
        improved_score = evaluation_func(improved)
        
        # Tenta apenas alguns ajustes aleatórios
        for _ in range(5):  # Limite pequeno para teste
            aa1 = random.choice(improved.aa_order)
            aa2 = random.choice(improved.aa_order)
            
            current = improved.get_score(aa1, aa2)
            adjustment = random.choice([-1, 1])
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

    def _shake(
        self,
        matrix: AdaptiveMatrix,
        neighborhood: VNSStructure
    ) -> AdaptiveMatrix:
        """Shake sistemático na vizinhança k."""
        perturbed = matrix.copy()
        boundaries = self.scores.get_block_boundaries(neighborhood.min_score)

        if not boundaries:
            return perturbed

        changes = int(
            self.hyperparams['VNS']['PERTURBATION_SIZE'] *
            neighborhood.probability
        )

        for _ in range(changes):
            start, stop = random.choice(boundaries)
            aa1 = random.choice(perturbed.aa_order)
            aa2 = random.choice(perturbed.aa_order)

            current = perturbed.get_score(aa1, aa2)
            adjustment = random.choice(neighborhood.get_adjustment_range())
            new_score = current + adjustment

            if perturbed._validate_score(aa1, aa2, new_score):
                perturbed.update_score(aa1, aa2, new_score)

        return perturbed

    def _local_search(
        self,
        matrix: AdaptiveMatrix,
        evaluation_func: Callable,
        neighborhood: VNSStructure
    ) -> AdaptiveMatrix:
        """Busca local completa na vizinhança."""
        improved = matrix.copy()
        improved_score = evaluation_func(improved)

        boundaries = self.scores.get_block_boundaries(neighborhood.min_score)
        if not boundaries:
            return improved

        for start, stop in boundaries:
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