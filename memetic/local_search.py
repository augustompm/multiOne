import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from collections import defaultdict
import logging
import random
from typing import Dict, List, Set, Tuple, Optional
from .matrix import AdaptiveMatrix
import re  # Add import for regex in ScoreAccessLayer

class ScoreAccessLayer:
    """
    Camada de abstração para acesso aos scores do alinhamento.
    Lida com diferentes nomenclaturas e fornece fallbacks.
    """
    def __init__(self, logger):
        self.column_scores = {}
        self.column_score_owners = {}
        self.group_score_mapping = {}
        self.logger = logger

    def load_from_xml(self, root) -> None:
        """Carrega scores do XML de forma resiliente."""
        self.column_scores.clear()
        self.column_score_owners.clear()
        self.group_score_mapping.clear()

        # Coleta todos os column-scores
        for colsco in root.findall(".//column-score"):
            try:
                name = colsco.find("colsco-name").text
                owner = colsco.find("colsco-owner").text if colsco.find("colsco-owner") is not None else ""
                data_text = colsco.find("colsco-data").text if colsco.find("colsco-data") is not None else ""
                
                if not data_text.strip():
                    continue
                    
                data = [int(x) for x in data_text.split()]
                
                # Guarda score e owner
                self.column_scores[name] = data
                if owner:
                    self.column_score_owners[name] = owner
                    
                # Mapeia grupo se aplicável
                self._map_group_score(name, owner)
                
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Error loading score {name}: {e}")
                continue

    def _map_group_score(self, name: str, owner: str) -> None:
        """Mapeia scores de grupo para suas diferentes nomenclaturas."""
        # Padrões conhecidos de nomenclatura
        patterns = [
            (r"normd_(\d+)", "name"),
            (r"group_?(\d+)", "owner"),
            (r"group(\d+)", "name"),
        ]
        
        for pattern, source in patterns:
            match = re.search(pattern, name if source == "name" else owner)
            if match:
                try:
                    group_id = int(match.group(1))
                    self.group_score_mapping[group_id] = name
                except ValueError:
                    continue

    def get_global_scores(self) -> List[int]:
        """Obtém scores globais com fallbacks."""
        # Tenta nomenclaturas conhecidas
        known_names = ['normd_all', 'all', 'global', 'overall']
        
        # Primeiro tenta pelo nome direto
        for name in known_names:
            if name in self.column_scores:
                return self.column_scores[name]
                
        # Depois tenta pelo owner
        for name, owner in self.column_score_owners.items():
            if owner in known_names:
                return self.column_scores[name]
                
        # Se não encontrou, tenta identificar pelo padrão dos dados
        for name, scores in self.column_scores.items():
            # Se é um vetor do tamanho do alinhamento e tem valores típicos de conservação
            if isinstance(scores, list) and len(scores) > 0:
                mean_score = np.mean(scores)
                if 0 <= mean_score <= 100:  # Scores de conservação típicos
                    self.logger.info(f"Using {name} as global scores based on data pattern")
                    return scores
                
        # Último recurso: média dos scores de grupo
        all_group_scores = []
        for scores in self.column_scores.values():
            if isinstance(scores, list) and len(scores) > 0:
                all_group_scores.append(scores)
                
        if all_group_scores:
            self.logger.warning("Using mean of all group scores as fallback")
            return np.mean(all_group_scores, axis=0).astype(int).tolist()
            
        self.logger.warning("No usable global scores found")
        return []

    def get_group_scores(self, group_id: int) -> List[int]:
        """Obtém scores para um grupo específico com fallbacks."""
        # Primeiro tenta pelo mapeamento conhecido
        if group_id in self.group_score_mapping:
            return self.column_scores[self.group_score_mapping[group_id]]
            
        # Tenta padrões de nomenclatura conhecidos
        patterns = [
            f'normd_{group_id}',
            f'group_{group_id}',
            f'group{group_id}',
        ]
        
        for pattern in patterns:
            if pattern in self.column_scores:
                return self.column_scores[pattern]
                
        # Se não encontrou específico, usa global
        self.logger.warning(f"No specific scores for group {group_id}, using global")
        return self.get_global_scores()

class EnhancedLocalSearch:
    """
    Busca local aprimorada baseada na estrutura do BAliBASE4.
    Referência: BAliBASE4-RV100.pdf seção "Methods" - 'Current Challenges'
    """
    def __init__(self, matrix: AdaptiveMatrix, hyperparams: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.matrix = matrix
        self.hyperparams = hyperparams
        
        # Estruturas para análise do BAliBASE4 conforme seção "Methods"
        self.conserved_blocks = defaultdict(list)     # Blocos por score
        self.subfamily_blocks = defaultdict(list)     # Blocos por grupo
        self.group_patterns = defaultdict(dict)       # Padrões por grupo
        self.column_scores = {                        # Scores por coluna/grupo
            'all': [],
            'group1': [],
            'group2': []
        }
        
        # Tracking de busca
        self.best_matrix = matrix.copy()
        self.best_score = float('-inf')
        self.move_history = defaultdict(int)  # Histórico de movimentos
        self.stagnation_count = 0

        # Usa nova camada de acesso aos scores
        self.scores = ScoreAccessLayer(self.logger)
        
    def analyze_alignment(self, xml_path: Path) -> None:
        """
        Análise do alinhamento baseada no BAliBASE4.
        Com tratamento resiliente para diferentes nomenclaturas de scores.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extrai scores dos blocos
            for block in root.findall(".//fitem/[ftype='BLOCK']"):
                start = int(block.find("fstart").text)
                stop = int(block.find("fstop").text)
                score = float(block.find("fscore").text)
                color = int(block.find("fcolor").text)
                
                block_data = {
                    'start': start,
                    'stop': stop,
                    'score': score,
                    'color': color,
                    'length': stop - start + 1
                }
                self.conserved_blocks[score].append(block_data)
            
            # Mapeia grupos e suas sequências
            self.sequence_groups = {}
            for seq in root.findall(".//sequence"):
                seq_name = seq.find("seq-name").text
                group = int(seq.find("seq-info/group").text)
                self.sequence_groups[seq_name] = group

            # Carrega scores usando nova camada
            self.scores.load_from_xml(root)
            
            self.logger.info(f"Analyzed BAliBASE4 data: {len(self.conserved_blocks)} blocks, "
                            f"{len(set(self.sequence_groups.values()))} groups")
            
        except Exception as e:
            self.logger.error(f"Error analyzing BAliBASE4 data: {e}")
            raise

    def get_group_scores(self, group_id: int) -> List[int]:
        """
        Obtém os scores para um grupo específico de forma resiliente.
        
        Args:
            group_id: ID do grupo
            
        Returns:
            Lista de scores do grupo ou scores gerais se não encontrar específicos
        """
        # Tenta diferentes padrões de nomenclatura
        possible_names = [
            f'group{group_id}',                # padrão original
            f'normd_{group_id}',              # padrão BBA0183
            f'group_{group_id}',              # outro possível padrão
        ]
        
        # Usa o mapeamento construído na análise
        if group_id in self.group_score_mapping:
            return self.column_scores[self.group_score_mapping[group_id]]
        
        # Tenta os padrões conhecidos
        for name in possible_names:
            if name in self.column_scores:
                return self.column_scores[name]
        
        # Se não encontrar scores específicos, usa o geral
        self.logger.warning(f"No specific scores found for group {group_id}, usando scores gerais")
        return self.column_scores.get('normd_all', [])

    def vns_search(self, evaluation_func, max_iterations: int, max_no_improve: int) -> float:
        """
        VNS com vizinhanças estruturais baseadas no BAliBASE4.
        Ref: BAliBASE4-RV100.pdf seção "Motifs in Disordered Regions"
        """
        current_matrix = self.matrix.copy()
        current_score = evaluation_func(current_matrix)
        
        self.best_score = current_score
        self.best_matrix = current_matrix.copy()
        
        iterations_no_improve = 0
        neighborhood_index = 0
        
        # Define vizinhanças estruturais baseadas no paper
        neighborhoods = [
            (self._conserved_blocks_neighborhood, "Conserved Blocks"),
            (self._subfamily_specific_neighborhood, "Subfamily Patterns"),
            (self._low_conservation_neighborhood, "Low Conservation Regions")
        ]
        
        self.logger.info("Starting VNS search with structural neighborhoods")
        
        while iterations_no_improve < max_no_improve:
            neighborhood_func, name = neighborhoods[neighborhood_index]
            
            # Aplica vizinhança com força adaptativa
            force_factor = min(1.5, 1 + (iterations_no_improve / max_no_improve))
            neighbor = neighborhood_func(current_matrix, force_factor)
            neighbor_score = evaluation_func(neighbor)
            
            # Atualiza melhor solução
            if neighbor_score > current_score + self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                current_matrix = neighbor
                current_score = neighbor_score
                
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_matrix = current_matrix.copy()
                    self.logger.info(f"New best score {current_score:.4f} found in {name}")
                    iterations_no_improve = 0
                    continue
                    
            # Muda vizinhança ou aplica perturbação
            neighborhood_index = (neighborhood_index + 1) % len(neighborhoods)
            iterations_no_improve += 1
            
            # Perturbação adaptativa em estagnação
            if iterations_no_improve > max_no_improve // 2:
                current_matrix = self._adaptive_perturbation(
                    current_matrix,
                    force_factor
                )
                current_score = evaluation_func(current_matrix)
                
        return self.best_score
        
    def _conserved_blocks_neighborhood(
        self, 
        matrix: AdaptiveMatrix,
        force_factor: float = 1.0
    ) -> AdaptiveMatrix:
        """
        Vizinhança baseada em blocos conservados.
        Ref: BAliBASE4-RV100.pdf seção "Conserved Blocks"
        """
        new_matrix = matrix.copy()
        
        # Seleciona blocos mais conservados
        high_scores = sorted(
            self.conserved_blocks.keys(),
            reverse=True
        )[:self.hyperparams['LOCAL_SEARCH']['HIGH_SCORE_BLOCKS']]
        
        for score in high_scores:
            blocks = self.conserved_blocks[score]
            samples = min(
                len(blocks),
                self.hyperparams['LOCAL_SEARCH']['SAMPLES_PER_BLOCK']
            )
            
            for block in random.sample(blocks, samples):
                # Ajusta scores proporcionalmente ao score do bloco
                adjustment_base = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']
                
                if score > self.hyperparams['LOCAL_SEARCH']['BLOCK_SCORE_THRESHOLD']:
                    base = adjustment_base['strong']
                else:
                    base = adjustment_base['medium']
                    
                # Aplica força adaptativa
                adjustment = int(base * force_factor)
                
                # Atualiza scores
                for aa1 in new_matrix.aa_order:
                    for aa2 in new_matrix.aa_order:
                        current = new_matrix.get_score(aa1, aa2)
                        new_score = current + adjustment
                        
                        if new_matrix._validate_score(aa1, aa2, new_score):
                            new_matrix.update_score(aa1, aa2, new_score)
                            
        return new_matrix
        
    def _subfamily_specific_neighborhood(
        self, 
        matrix: AdaptiveMatrix,
        force_factor: float = 1.0
    ) -> AdaptiveMatrix:
        """
        Vizinhança baseada em padrões específicos de subfamílias.
        Usa tratamento resiliente para scores por grupo.
        """
        new_matrix = matrix.copy()
        
        # Para cada subfamília
        for group_id, blocks in self.subfamily_blocks.items():
            # Usa tratamento resiliente para obter scores do grupo
            group_scores = self.get_group_scores(group_id)
            
            if not group_scores:
                continue
                
            # Identifica posições conservadas no grupo
            conserved_positions = [
                i for i, score in enumerate(group_scores)
                if score > np.mean(group_scores) + np.std(group_scores)
            ]
            
            # Ajusta scores
            changes = self.hyperparams['LOCAL_SEARCH']['SUBFAMILY_CHANGES']
            adjustment_base = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']
            
            for _ in range(changes):
                if conserved_positions:
                    pos = random.choice(conserved_positions)
                    score = group_scores[pos]
                    
                    # Ajuste proporcional ao score
                    if score > np.percentile(group_scores, 75):
                        base = adjustment_base['medium']
                    else:
                        base = adjustment_base['weak']
                        
                    # Aplica força adaptativa
                    adjustment = int(base * force_factor)
                    
                    # Atualiza matriz
                    for aa1 in new_matrix.aa_order:
                        for aa2 in new_matrix.aa_order:
                            current = new_matrix.get_score(aa1, aa2)
                            new_score = current + adjustment
                            
                            if new_matrix._validate_score(aa1, aa2, new_score):
                                new_matrix.update_score(aa1, aa2, new_score)
                                
        return new_matrix
        
    def _low_conservation_neighborhood(
        self, 
        matrix: AdaptiveMatrix,
        force_factor: float = 1.0
    ) -> AdaptiveMatrix:
        """
        Vizinhança para regiões menos conservadas.
        Ref: BAliBASE4-RV100.pdf seção "Low Conservation Regions"
        """
        new_matrix = matrix.copy()
        
        # Usa nova camada para obter scores
        all_scores = self.scores.get_global_scores()
        
        if not all_scores:
            return new_matrix
            
        # Resto da implementação igual...
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        low_conservation = [
            i for i, score in enumerate(all_scores)
            if score < mean_score - std_score
        ]
        
        changes = self.hyperparams['LOCAL_SEARCH']['CONSERVATION_TOP_N']
        adjustment_base = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']
        
        for pos in random.sample(low_conservation, min(changes, len(low_conservation))):
            score = all_scores[pos]
            
            if score < np.percentile(all_scores, 25):
                base = adjustment_base['weak'] * -1
            else:
                base = adjustment_base['medium'] * -1
                
            adjustment = int(base * force_factor)
            
            for aa1 in new_matrix.aa_order:
                for aa2 in new_matrix.aa_order:
                    current = new_matrix.get_score(aa1, aa2)
                    new_score = current + adjustment
                    
                    if new_matrix._validate_score(aa1, aa2, new_score):
                        new_matrix.update_score(aa1, aa2, new_score)
                        
        return new_matrix
        
    def _adaptive_perturbation(
        self, 
        matrix: AdaptiveMatrix,
        force_factor: float = 1.0
    ) -> AdaptiveMatrix:
        """
        Perturbação adaptativa baseada no histórico.
        Aumenta força em áreas promissoras.
        """
        perturbed = matrix.copy()
        size = int(self.hyperparams['VNS']['PERTURBATION_SIZE'] * force_factor)
        
        # Usa histórico para guiar perturbação
        promising_pairs = sorted(
            self.move_history.items(),
            key=lambda x: x[1],
            reverse=True
        )[:size]
        
        # Aplica perturbações direcionadas
        for (aa1, aa2), _ in promising_pairs:
            current = perturbed.get_score(aa1, aa2)
            
            # Perturbação mais forte em estagnação
            adjustment = random.choice([-3, -2, -1, 1, 2]) * force_factor
            new_score = current + int(adjustment)
            
            if perturbed._validate_score(aa1, aa2, new_score):
                perturbed.update_score(aa1, aa2, new_score)
                self.move_history[(aa1, aa2)] += 1
                
        return perturbed