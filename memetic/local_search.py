import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from collections import defaultdict
import logging
import random
from typing import Dict, List, Set, Tuple, Optional

from .matrix import AdaptiveMatrix

class EnhancedLocalSearch:
    def __init__(self, matrix: AdaptiveMatrix, hyperparams: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.matrix = matrix
        self.hyperparams = hyperparams
        
        # Estruturas para análise do BAliBASE4
        self.core_blocks = defaultdict(list)  # Blocos por cor (nível de conservação)
        self.disorder_regions = []  # Regiões com DISORDER tag
        self.subfamily_groups = defaultdict(list)  # Sequências por grupo
        self.conservation_scores = defaultdict(float)  # Scores por posição
        
        self.best_matrix = matrix.copy()
        self.best_score = float('-inf')
        
    def analyze_alignment(self, xml_path: Path) -> None:
        """Análise detalhada do alinhamento BAliBASE4."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for seq in root.findall(".//sequence"):
                seq_name = seq.find("seq-name").text
                seq_data = seq.find("seq-data").text.strip()
                group = int(seq.find("seq-info/group").text)
                
                # Agrupa sequências por subfamília
                self.subfamily_groups[group].append({
                    'name': seq_name,
                    'data': seq_data
                })
                
                # Processa blocos conservados (core blocks)
                for block in seq.findall(".//fitem/[ftype='BLOCK']"):
                    color = int(block.find("fcolor").text)  # Nível de conservação
                    start = int(block.find("fstart").text)
                    stop = int(block.find("fstop").text)
                    score = float(block.find("fscore").text)
                    
                    self.core_blocks[color].append({
                        'seq': seq_name,
                        'group': group,
                        'start': start,
                        'stop': stop,
                        'data': seq_data[start-1:stop],
                        'score': score
                    })
                    
                    # Atualiza scores de conservação por posição
                    for pos in range(start-1, stop):
                        if seq_data[pos] != '-':
                            self.conservation_scores[pos] += score
                            
                # Registra regiões desordenadas
                for disorder in seq.findall(".//fitem/[ftype='DISORDER']"):
                    start = int(disorder.find("fstart").text)
                    stop = int(disorder.find("fstop").text)
                    
                    self.disorder_regions.append({
                        'seq': seq_name,
                        'start': start,
                        'stop': stop,
                        'data': seq_data[start-1:stop]
                    })
                    
            self._analyze_subfamily_patterns()
            self._normalize_conservation_scores()
            
        except Exception as e:
            self.logger.error(f"Error analyzing BAliBASE4 data: {e}")
            raise
            
    def _analyze_subfamily_patterns(self) -> None:
        """Analisa padrões específicos de cada subfamília."""
        self.subfamily_patterns = defaultdict(lambda: defaultdict(float))
        
        for group, blocks in self.core_blocks.items():
            high_conserved = [b for b in blocks if b['score'] > 0.8]
            
            for block in high_conserved:
                data = block['data']
                score = block['score']
                subfamily = block['group']
                
                # Analisa padrões de substituição conservados
                for i in range(len(data)-1):
                    if data[i] != '-' and data[i+1] != '-':
                        pair = tuple(sorted([data[i], data[i+1]]))
                        self.subfamily_patterns[subfamily][pair] += score
                        
    def _normalize_conservation_scores(self) -> None:
        """Normaliza scores de conservação."""
        if self.conservation_scores:
            max_score = max(self.conservation_scores.values())
            if max_score > 0:
                for pos in self.conservation_scores:
                    self.conservation_scores[pos] /= max_score
                    
    def vns_search(self, evaluation_func, max_iterations: int, max_no_improve: int) -> float:
        """VNS-ILS baseado em evidências do BAliBASE4."""
        current_matrix = self.matrix.copy()
        current_score = evaluation_func(current_matrix)
        
        self.best_score = current_score
        self.best_matrix = current_matrix.copy()
        
        iterations_no_improve = 0
        
        while iterations_no_improve < max_no_improve:
            improved = False
            
            # Vizinhanças baseadas em evidências do BAliBASE4
            neighborhoods = [
                (self._core_block_neighborhood, 0.4),      # Core blocks altamente conservados
                (self._subfamily_specific_neighborhood, 0.3),  # Padrões específicos
                (self._disorder_aware_neighborhood, 0.2),   # Regiões desordenadas
                (self._random_neighborhood, 0.1)           # Exploração
            ]
            
            for neighborhood, prob in neighborhoods:
                if random.random() < prob:
                    neighbor = neighborhood(current_matrix)
                    neighbor_score = evaluation_func(neighbor)
                    
                    if neighbor_score > current_score + self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                        current_matrix = neighbor
                        current_score = neighbor_score
                        improved = True
                        
                        if current_score > self.best_score:
                            self.best_score = current_score
                            self.best_matrix = current_matrix.copy()
                            iterations_no_improve = 0
                            break
                            
            if not improved:
                iterations_no_improve += 1
                if iterations_no_improve > max_no_improve // 2:
                    current_matrix = self._adaptive_perturbation(current_matrix)
                    current_score = evaluation_func(current_matrix)
                    
        return self.best_score
        
    def _core_block_neighborhood(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Vizinhança baseada em core blocks altamente conservados."""
        new_matrix = matrix.copy()
        
        # Seleciona blocos altamente conservados
        high_conserved = []
        for blocks in self.core_blocks.values():
            high_conserved.extend([b for b in blocks if b['score'] > 0.8])
            
        if high_conserved:
            block = random.choice(high_conserved)
            data = block['data']
            
            # Ajusta scores para substituições observadas
            for i in range(len(data)-1):
                if data[i] != '-' and data[i+1] != '-':
                    aa1, aa2 = data[i], data[i+1]
                    current = new_matrix.get_score(aa1, aa2)
                    
                    # Minimização mais forte para padrões conservados
                    new_score = current + self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['strong']
                    
                    if new_matrix._validate_score(aa1, aa2, new_score):
                        new_matrix.update_score(aa1, aa2, new_score)
                        
        return new_matrix
        
    def _subfamily_specific_neighborhood(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Vizinhança baseada em padrões específicos de subfamílias."""
        new_matrix = matrix.copy()
        
        if self.subfamily_patterns:
            # Seleciona subfamília aleatória
            subfamily = random.choice(list(self.subfamily_patterns.keys()))
            patterns = self.subfamily_patterns[subfamily]
            
            # Pega top N substituições mais frequentes
            top_n = self.hyperparams['LOCAL_SEARCH']['SUBFAMILY_CHANGES']
            top_pairs = sorted(
                patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            for (aa1, aa2), weight in top_pairs:
                current = new_matrix.get_score(aa1, aa2)
                
                # Ajuste baseado no peso do padrão
                if weight > 0.5:
                    adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['strong']
                else:
                    adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['medium']
                    
                new_score = current + adjustment
                if new_matrix._validate_score(aa1, aa2, new_score):
                    new_matrix.update_score(aa1, aa2, new_score)
                    
        return new_matrix
        
    def _disorder_aware_neighborhood(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Vizinhança especializada para regiões desordenadas."""
        new_matrix = matrix.copy()
        
        if not self.hyperparams['LOCAL_SEARCH']['USE_DISORDER_INFO']:
            return new_matrix
            
        # Analisa padrões em regiões desordenadas
        disorder_patterns = defaultdict(float)
        for region in self.disorder_regions:
            data = region['data']
            for i in range(len(data)-1):
                if data[i] != '-' and data[i+1] != '-':
                    pair = tuple(sorted([data[i], data[i+1]]))
                    disorder_patterns[pair] += 1
                    
        if disorder_patterns:
            max_changes = self.hyperparams['LOCAL_SEARCH']['DISORDER_CHANGES']
            patterns = random.sample(
                list(disorder_patterns.items()),
                min(max_changes, len(disorder_patterns))
            )
            
            for (aa1, aa2), freq in patterns:
                current = new_matrix.get_score(aa1, aa2)
                
                # Ajustes mais suaves para regiões desordenadas
                if freq > 2:
                    adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['medium']
                else:
                    adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['weak']
                    
                new_score = current + adjustment
                if new_matrix._validate_score(aa1, aa2, new_score):
                    new_matrix.update_score(aa1, aa2, new_score)
                    
        return new_matrix
        
    def _random_neighborhood(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Vizinhança aleatória para exploração."""
        new_matrix = matrix.copy()
        
        num_changes = self.hyperparams['LOCAL_SEARCH']['RANDOM_CHANGES']
        for _ in range(num_changes):
            aa1 = random.choice(new_matrix.aa_order)
            aa2 = random.choice(new_matrix.aa_order)
            current = new_matrix.get_score(aa1, aa2)
            
            adjustment = random.choice([
                self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['strong'],
                self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['medium'],
                self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['weak']
            ])
            
            new_score = current + adjustment
            if new_matrix._validate_score(aa1, aa2, new_score):
                new_matrix.update_score(aa1, aa2, new_score)
                
        return new_matrix
        
    def _adaptive_perturbation(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Perturbação adaptativa baseada em evidências."""
        perturbed = matrix.copy()
        size = self.hyperparams['VNS']['PERTURBATION_SIZE']
        
        neighborhoods = [
            (self._core_block_neighborhood, 0.4),
            (self._subfamily_specific_neighborhood, 0.3),
            (self._disorder_aware_neighborhood, 0.2),
            (self._random_neighborhood, 0.1)
        ]
        
        for _ in range(size):
            strategy = random.choices(
                [n[0] for n in neighborhoods],
                weights=[n[1] for n in neighborhoods]
            )[0]
            perturbed = strategy(perturbed)
            
        return perturbed