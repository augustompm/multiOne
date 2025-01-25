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
        
        # Estruturas para análise detalhada do BAliBASE4
        self.conserved_blocks = defaultdict(list)     # Blocos por score
        self.subfamily_blocks = defaultdict(list)     # Blocos por subfamília
        self.conservation_patterns = defaultdict(float)  # Padrões conservados
        self.conservation_weights = defaultdict(float)   # Pesos por posição
        self.group_patterns = defaultdict(dict)       # Padrões específicos por grupo
        self.disorder_regions = []                    # Regiões desordenadas
        
        # Tracking de resultados
        self.best_matrix = matrix.copy()
        self.best_score = float('-inf')
        
    def analyze_alignment(self, xml_path: Path) -> None:
        """Análise abrangente do alinhamento BAliBASE4."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            sequences = {}
            for seq in root.findall(".//sequence"):
                seq_name = seq.find("seq-name").text
                seq_data = seq.find("seq-data").text.strip()
                group = int(seq.find("seq-info/group").text)
                sequences[seq_name] = {'data': seq_data, 'group': group}
                
                # Processa blocos conservados
                for block in seq.findall(".//fitem/[ftype='BLOCK']"):
                    start = int(block.find("fstart").text)
                    stop = int(block.find("fstop").text)
                    score = float(block.find("fscore").text)
                    color = int(block.find("fcolor").text)
                    
                    block_data = {
                        'seq': seq_name,
                        'group': group,
                        'start': start,
                        'stop': stop,
                        'data': seq_data[start-1:stop],
                        'score': score,
                        'color': color
                    }
                    
                    self.conserved_blocks[score].append(block_data)
                    self.subfamily_blocks[group].append(block_data)
                    
                # Processa regiões desordenadas
                for disorder in seq.findall(".//fitem/[ftype='DISORDER']"):
                    start = int(disorder.find("fstart").text)
                    stop = int(disorder.find("fstop").text)
                    
                    self.disorder_regions.append({
                        'seq': seq_name,
                        'group': group,
                        'start': start,
                        'stop': stop,
                        'data': seq_data[start-1:stop]
                    })
            
            # Analisa padrões de conservação
            self._analyze_conservation_patterns()
            # Analisa padrões específicos de grupos
            self._analyze_group_patterns(sequences)
            # Normaliza pesos de conservação
            self._normalize_conservation_weights()
            
        except Exception as e:
            self.logger.error(f"Error analyzing BAliBASE4 data: {e}")
            raise
            
    def _analyze_conservation_patterns(self) -> None:
        """Analisa padrões de conservação em blocos."""
        for blocks in self.subfamily_blocks.values():
            for block in blocks:
                data = block['data']
                score = block['score']
                
                # Analisa substituições em janelas
                window = self.hyperparams['LOCAL_SEARCH'].get('PATTERN_WINDOW', 2)
                for i in range(len(data) - window + 1):
                    pattern = data[i:i+window]
                    if '-' not in pattern:
                        self.conservation_patterns[pattern] += score
                        
    def _analyze_group_patterns(self, sequences: Dict) -> None:
        """Analisa padrões específicos de cada grupo."""
        for group_id in set(seq['group'] for seq in sequences.values()):
            group_seqs = {name: seq for name, seq in sequences.items() 
                         if seq['group'] == group_id}
            
            patterns = defaultdict(float)
            for block in self.subfamily_blocks[group_id]:
                data = block['data']
                score = block['score']
                
                for i in range(len(data)-1):
                    if data[i] != '-' and data[i+1] != '-':
                        pair = tuple(sorted([data[i], data[i+1]]))
                        patterns[pair] += score
                        
            self.group_patterns[group_id] = patterns
            
    def _normalize_conservation_weights(self) -> None:
        """Normaliza pesos de conservação."""
        if self.conservation_patterns:
            max_weight = max(self.conservation_patterns.values())
            if max_weight > 0:
                for pattern in self.conservation_patterns:
                    self.conservation_patterns[pattern] /= max_weight
                    
    def vns_search(self, evaluation_func, max_iterations: int, max_no_improve: int) -> float:
        """VNS com três vizinhanças estruturais e tracking detalhado."""
        current_matrix = self.matrix.copy()
        current_score = evaluation_func(current_matrix)
        
        self.best_score = current_score
        self.best_matrix = current_matrix.copy()
        
        iterations_no_improve = 0
        neighborhood_index = 0
        
        # Define vizinhanças estruturais
        neighborhoods = [
            (self._conserved_blocks_neighborhood, "Conserved Blocks"),
            (self._group_specific_neighborhood, "Group Patterns"),
            (self._disorder_aware_neighborhood, "Disorder Regions")
        ]
        
        self.logger.info("Starting VNS search with structural neighborhoods")
        
        while iterations_no_improve < max_no_improve:
            neighborhood_func, neighborhood_name = neighborhoods[neighborhood_index]
            
            # Tenta vizinhança atual
            neighbor = neighborhood_func(current_matrix)
            neighbor_score = evaluation_func(neighbor)
            
            if neighbor_score > current_score + self.hyperparams['VNS']['MIN_IMPROVEMENT']:
                current_matrix = neighbor
                current_score = neighbor_score
                
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_matrix = current_matrix.copy()
                    self.logger.info(f"New best score {current_score:.4f} found in {neighborhood_name}")
                    iterations_no_improve = 0
                    continue
                    
            # Muda vizinhança se não melhorou
            neighborhood_index = (neighborhood_index + 1) % len(neighborhoods)
            iterations_no_improve += 1
            
            # Perturbação adaptativa
            if iterations_no_improve > max_no_improve // 2:
                current_matrix = self._adaptive_perturbation(current_matrix)
                current_score = evaluation_func(current_matrix)
                
        return self.best_score
        
    def _conserved_blocks_neighborhood(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Vizinhança baseada em blocos altamente conservados."""
        new_matrix = matrix.copy()
        
        # Seleciona blocos mais conservados
        high_scores = sorted(self.conserved_blocks.keys(), reverse=True)
        top_n = self.hyperparams['LOCAL_SEARCH'].get('HIGH_SCORE_BLOCKS', 3)
        
        for score in high_scores[:top_n]:
            blocks = self.conserved_blocks[score]
            samples = self.hyperparams['LOCAL_SEARCH'].get('SAMPLES_PER_BLOCK', 2)
            
            for block in random.sample(blocks, min(samples, len(blocks))):
                data = block['data']
                
                # Ajusta scores baseado no padrão de conservação
                for i in range(len(data)-1):
                    if data[i] != '-' and data[i+1] != '-':
                        aa1, aa2 = data[i], data[i+1]
                        current = new_matrix.get_score(aa1, aa2)
                        
                        # Ajuste proporcional ao score do bloco
                        threshold = self.hyperparams['LOCAL_SEARCH'].get('BLOCK_SCORE_THRESHOLD', 25)
                        if score > threshold:
                            adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['strong']
                        else:
                            adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['medium']
                            
                        new_score = current + adjustment
                        if new_matrix._validate_score(aa1, aa2, new_score):
                            new_matrix.update_score(aa1, aa2, new_score)
                            
        return new_matrix
        
    def _group_specific_neighborhood(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Vizinhança baseada em padrões específicos de grupos."""
        new_matrix = matrix.copy()
        
        # Seleciona grupo aleatoriamente
        group = random.choice(list(self.group_patterns.keys()))
        patterns = self.group_patterns[group]
        
        # Ajusta scores para padrões frequentes
        changes = self.hyperparams['LOCAL_SEARCH']['SUBFAMILY_CHANGES']
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:changes]
        
        for (aa1, aa2), weight in top_patterns:
            current = new_matrix.get_score(aa1, aa2)
            
            # Ajuste baseado no peso do padrão
            if weight > 0.5:  # Padrão muito frequente no grupo
                adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['medium']
            else:
                adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['weak']
                
            new_score = current + adjustment
            if new_matrix._validate_score(aa1, aa2, new_score):
                new_matrix.update_score(aa1, aa2, new_score)
                
        return new_matrix
        
    def _disorder_aware_neighborhood(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Vizinhança especializada para regiões desordenadas."""
        new_matrix = matrix.copy()
        
        if not self.disorder_regions or not self.hyperparams['LOCAL_SEARCH']['USE_DISORDER_INFO']:
            return new_matrix
            
        # Analisa padrões em regiões desordenadas
        disorder_patterns = defaultdict(float)
        for region in self.disorder_regions:
            data = region['data']
            for i in range(len(data)-1):
                if data[i] != '-' and data[i+1] != '-':
                    pair = tuple(sorted([data[i], data[i+1]]))
                    disorder_patterns[pair] += 1
                    
        # Ajusta scores permitindo mais flexibilidade
        changes = self.hyperparams['LOCAL_SEARCH']['DISORDER_CHANGES']
        patterns = random.sample(
            list(disorder_patterns.items()),
            min(changes, len(disorder_patterns))
        )
        
        for (aa1, aa2), freq in patterns:
            current = new_matrix.get_score(aa1, aa2)
            # Ajustes mais flexíveis para regiões desordenadas
            adjustment = self.hyperparams['LOCAL_SEARCH']['SCORE_ADJUSTMENTS']['weak']
            new_score = current + adjustment
            
            if new_matrix._validate_score(aa1, aa2, new_score):
                new_matrix.update_score(aa1, aa2, new_score)
                
        return new_matrix
        
    def _adaptive_perturbation(self, matrix: AdaptiveMatrix) -> AdaptiveMatrix:
        """Perturbação que preserva estrutura do alinhamento."""
        perturbed = matrix.copy()
        size = self.hyperparams['VNS']['PERTURBATION_SIZE']
        
        # Combina diferentes estratégias de perturbação
        for _ in range(size):
            method = random.choice([
                self._conserved_blocks_neighborhood,
                self._group_specific_neighborhood,
                self._disorder_aware_neighborhood
            ])
            perturbed = method(perturbed)
            
        return perturbed