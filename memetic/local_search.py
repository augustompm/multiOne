# memetic/local_search.py

import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict
import logging
import random
from datetime import datetime
from .matrix import AdaptiveMatrix

class LocalSearch:
    """
    Variable Neighborhood Search implementation for matrix optimization using
    three biologically-motivated neighborhood structures, enhanced with Iterated
    Local Search (ILS) strategies.
    """
    
    def __init__(self, matrix: AdaptiveMatrix,
                 min_improvement: float,
                 perturbation_size: int,
                 max_perturbation: int,
                 score_constraints: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Inicializando a classe LocalSearch.")
        
        self.matrix = matrix
        self.logger.debug("Matriz adaptativa inicializada.")
        
        self.best_matrix = matrix.copy()
        self.current_matrix = matrix.copy()
        self.logger.debug("Cópias de melhor e matriz atual inicializadas.")
        
        # Parâmetros do VNS-ILS vindos do HYPERPARAMS global
        self.min_improvement = min_improvement
        self.perturbation_size = perturbation_size  # Número inicial de mudanças
        self.max_perturbation = max_perturbation    # Limite máximo de perturbação
        self.logger.debug(f"Parâmetros de perturbação inicializados: min_improvement={self.min_improvement}, "
                          f"perturbation_size={self.perturbation_size}, max_perturbation={self.max_perturbation}.")
        
        # Estruturas para análise
        self.substitution_frequencies = defaultdict(lambda: defaultdict(float))
        self.logger.debug("Frequências de substituição inicializadas.")
        
        self.conservation_weights = defaultdict(float)
        self.logger.debug("Pesos de conservação inicializados.")
        
        self.best_score = float('-inf')
        self.logger.debug(f"Melhor score inicializado para {self.best_score}.")
        
        # Parâmetros de qualidade
        self.max_attempts = 50
        self.max_no_improve = 30
        self.logger.debug(f"Parâmetros de qualidade: max_attempts={self.max_attempts}, max_no_improve={self.max_no_improve}.")
        
        # Track improvements and history
        self.improvements = []
        self.logger.debug("Lista de melhorias inicializada.")
        
        self.matrix_history = set()
        self.logger.debug("Histórico de matrizes inicializado.")
        
        # Track neighborhood performance
        self.neighborhood_stats = {
            'frequency': {'attempts': 0, 'improvements': 0},
            'conservation': {'attempts': 0, 'improvements': 0},
            'group': {'attempts': 0, 'improvements': 0}
        }
        self.logger.debug("Estatísticas de vizinhança inicializadas.")
        
        # Amino acid groups based on physicochemical properties
        self.aa_groups = {
            'hydrophobic': {'I', 'L', 'V', 'M', 'F', 'W', 'A'},
            'polar': {'S', 'T', 'N', 'Q'},
            'acidic': {'D', 'E'},
            'basic': {'K', 'R', 'H'},
            'special': {'C', 'G', 'P', 'Y'}
        }
        self.logger.debug(f"Grupos de aminoácidos definidos: {list(self.aa_groups.keys())}.")
        
        # Define valid score ranges per matrix region
        self.score_constraints = score_constraints
        self.logger.debug("Restrições de pontuação definidas para regiões da matriz.")
    
    def _are_similar(self, aa1: str, aa2: str) -> bool:
        """Verifica se dois aminoácidos pertencem ao mesmo grupo físico-químico."""
        return any(aa1 in group and aa2 in group for group in self.aa_groups.values())

    def _validate_score_change(self, aa1: str, aa2: str, new_score: float) -> bool:
        """Validação de mudanças considerando restrições biológicas"""
        if aa1 == aa2:
            constraints = self.score_constraints['diagonal']
        elif self._are_similar(aa1, aa2):
            constraints = self.score_constraints['similar']
        else:
            constraints = self.score_constraints['different']
        
        self.logger.debug(f"Validando mudança de score para {aa1}-{aa2}: {new_score} com intervalo {constraints}.")
        return constraints['min'] <= new_score <= constraints['max']

    def analyze_alignment(self, xml_path: Path) -> None:
        """Analyzes alignment to extract substitution frequencies and conservation weights."""
        self.logger.info(f"Iniciando análise de alinhamento no arquivo: {xml_path}.")
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            self.logger.debug("Arquivo XML parseado com sucesso.")
            
            sequences = {}
            blocks = []
            
            for seq in root.findall(".//sequence"):
                seq_name = seq.find("seq-name").text
                seq_data = seq.find("seq-data").text.strip()
                sequences[seq_name] = seq_data
                self.logger.debug(f"Sequência extraída: {seq_name} com {len(seq_data)} caracteres.")
                
                for block in seq.findall(".//fitem/[ftype='BLOCK']"):
                    start = int(block.find("fstart").text)
                    stop = int(block.find("fstop").text)
                    score = float(block.find("fscore").text)
                    
                    blocks.append({
                        'seq': seq_name,
                        'start': start,
                        'stop': stop,
                        'score': score,
                        'data': seq_data[start-1:stop]
                    })
                    self.logger.debug(f"Bloco extraído: {blocks[-1]}.")
            
            self._analyze_patterns(sequences, blocks)
            self.logger.info("Análise de padrões concluída.")
            
        except Exception as e:
            self.logger.error(f"Error analyzing alignment: {e}")
            raise

    def _analyze_patterns(self, sequences: Dict[str, str], blocks: List[Dict]) -> None:
        """Analyzes substitution patterns and conservation in blocks."""
        self.logger.info("Analisando padrões de conservação...")
        self.substitution_frequencies.clear()
        self.conservation_weights.clear()
        self.logger.debug("Frequências de substituição e pesos de conservação resetados.")
        
        # Group blocks by position
        aligned_blocks = defaultdict(list)
        for block in blocks:
            aligned_blocks[block['start']].append(block)
        self.logger.debug(f"Blocos agrupados por posição alinhada: {len(aligned_blocks)} grupos.")
        
        # Analyze each aligned position
        for pos, block_group in aligned_blocks.items():
            avg_score = np.mean([b['score'] for b in block_group])
            
            for i, block1 in enumerate(block_group):
                for j, block2 in enumerate(block_group[i+1:], i+1):
                    for offset in range(min(len(block1['data']), len(block2['data']))):
                        aa1 = block1['data'][offset]
                        aa2 = block2['data'][offset]
                        
                        if aa1 != '-' and aa2 != '-':
                            weight = avg_score * (1.0 if aa1 != aa2 else 0.5)
                            self.substitution_frequencies[aa1][aa2] += weight
                            self.conservation_weights[aa1] += weight
                            self.conservation_weights[aa2] += weight
                            # Reduzido o logging excessivo
                            
        self.logger.info("Análise de padrões concluída.")
    
    def _store_improvement(self, score_improvement: float, neighborhood_index: int):
        """Registra melhoria de forma centralizada"""
        self.improvements.append({
            'score': score_improvement,
            'neighborhood': neighborhood_index,
            'perturbation_size': self.perturbation_size
        })
        self.logger.debug(f"Melhoria registrada: score={score_improvement}, "
                          f"neighborhood={neighborhood_index}, "
                          f"perturbation_size={self.perturbation_size}.")
    
    def perturb_solution(self) -> AdaptiveMatrix:
        """ILS: Aplica perturbação controlada na solução atual."""
        self.logger.debug("Aplicando perturbação na solução atual.")
        perturbed = self.current_matrix.copy()
        positions = list(range(len(self.matrix.aa_order)))
        
        for _ in range(self.perturbation_size):
            i, j = random.sample(positions, 2)
            aa1, aa2 = self.matrix.aa_order[i], self.matrix.aa_order[j]
            current_score = perturbed.get_score(aa1, aa2)
            
            # Perturbação baseada em grupos
            if self._are_similar(aa1, aa2):
                adjustment = random.choice([-1, 1])
            else:
                adjustment = random.choice([-2, -1, 1, 2])
                
            new_score = current_score + adjustment
            if self._validate_score_change(aa1, aa2, new_score):
                perturbed.update_score(aa1, aa2, new_score)
                self.logger.debug(f"Perturbação aplicada: {aa1}-{aa2} de {current_score} para {new_score}.")
            else:
                self.logger.debug(f"Perturbação rejeitada para {aa1}-{aa2}: novo score {new_score} fora das restrições.")
        
        return perturbed

    def _frequency_based_neighborhood(self) -> AdaptiveMatrix:
        """Vizinhança baseada nas frequências de substituição."""
        self.logger.debug("Gerando vizinhança baseada em frequência.")
        new_matrix = self.current_matrix.copy()
        self.neighborhood_stats['frequency']['attempts'] += 1
        
        # Seleciona top substituições
        sorted_subs = [
            (aa1, aa2, freq) 
            for aa1, subs in self.substitution_frequencies.items()
            for aa2, freq in subs.items()
        ]
        sorted_subs.sort(key=lambda x: x[2], reverse=True)
        self.logger.debug(f"Top substituições por frequência selecionadas: {sorted_subs[:5]}")
        
        # Modifica top pares mais frequentes
        improvements = 0
        for aa1, aa2, freq in sorted_subs[:5]:
            current = new_matrix.get_score(aa1, aa2)
            median_freq = np.median([x[2] for x in sorted_subs]) if sorted_subs else 1
            direction = 1 if freq > median_freq else -1
            new_score = current + direction
            
            if self._validate_score_change(aa1, aa2, new_score):
                new_matrix.update_score(aa1, aa2, new_score)
                improvements += 1
                self.logger.debug(f"Vizinhança por frequência: {aa1}-{aa2} ajustado de {current} para {new_score}.")
            else:
                self.logger.debug(f"Ajuste por frequência rejeitado para {aa1}-{aa2}: novo score {new_score} fora das restrições.")
        
        if improvements > 0:
            self.neighborhood_stats['frequency']['improvements'] += 1
        
        return new_matrix

    def _conservation_based_neighborhood(self) -> AdaptiveMatrix:
        """Vizinhança baseada nos padrões de conservação."""
        self.logger.debug("Gerando vizinhança baseada em conservação.")
        new_matrix = self.current_matrix.copy()
        self.neighborhood_stats['conservation']['attempts'] += 1
        
        # Seleciona aminoácidos mais conservados
        conserved = sorted(self.conservation_weights.items(), 
                           key=lambda x: x[1], reverse=True)[:4]
        self.logger.debug(f"Aminoácidos mais conservados selecionados: {conserved}")
        
        improvements = 0
        for aa1, weight1 in conserved:
            # Fortalece relações entre conservados
            for aa2, weight2 in conserved:
                if aa1 != aa2:
                    current = new_matrix.get_score(aa1, aa2)
                    new_score = current + 1
                    if self._validate_score_change(aa1, aa2, new_score):
                        new_matrix.update_score(aa1, aa2, new_score)
                        improvements += 1
                        self.logger.debug(f"Conservação: {aa1}-{aa2} ajustado de {current} para {new_score}.")
                    else:
                        self.logger.debug(f"Ajuste de conservação rejeitado para {aa1}-{aa2}: novo score {new_score} fora das restrições.")
            
            # Diminui scores com não conservados
            for aa2 in self.matrix.aa_order:
                if aa2 not in [aa for aa, _ in conserved]:
                    current = new_matrix.get_score(aa1, aa2)
                    new_score = current - 1
                    if self._validate_score_change(aa1, aa2, new_score):
                        new_matrix.update_score(aa1, aa2, new_score)
                        improvements += 1
                        self.logger.debug(f"Conservação: {aa1}-{aa2} ajustado de {current} para {new_score}.")
                    else:
                        self.logger.debug(f"Ajuste de conservação rejeitado para {aa1}-{aa2}: novo score {new_score} fora das restrições.")
        
        if improvements > 0:
            self.neighborhood_stats['conservation']['improvements'] += 1
        
        return new_matrix

    def _group_based_neighborhood(self) -> AdaptiveMatrix:
        """Vizinhança baseada em grupos físico-químicos."""
        self.logger.debug("Gerando vizinhança baseada em grupos físico-químicos.")
        new_matrix = self.current_matrix.copy()
        self.neighborhood_stats['group']['attempts'] += 1
        
        # Seleciona grupo aleatório
        group_aas = random.choice(list(self.aa_groups.values()))
        group_list = list(group_aas)
        self.logger.debug(f"Grupo selecionado para ajuste: {group_list}")
        
        improvements = 0
        for i, aa1 in enumerate(group_list):
            for aa2 in group_list[i+1:]:
                current = new_matrix.get_score(aa1, aa2)
                new_score = current + 1
                if self._validate_score_change(aa1, aa2, new_score):
                    new_matrix.update_score(aa1, aa2, new_score)
                    improvements += 1
                    self.logger.debug(f"Grupo físico-químico: {aa1}-{aa2} ajustado de {current} para {new_score}.")
                else:
                    self.logger.debug(f"Ajuste de grupo rejeitado para {aa1}-{aa2}: novo score {new_score} fora das restrições.")
        
        if improvements > 0:
            self.neighborhood_stats['group']['improvements'] += 1
        
        return new_matrix

    def vns_search(self, evaluation_func, max_iterations: int = 100,
                  max_no_improve: int = 20) -> float:
        """VNS com ILS integrado."""
        self.logger.info(f"Iniciando busca VNS-ILS com max_iterations={max_iterations}, max_no_improve={max_no_improve}")
        
        current_score = evaluation_func(self.current_matrix)
        self.best_score = current_score
        iterations_no_improve = 0
        
        while iterations_no_improve < max_no_improve:
            self.logger.debug(f"Iteração {iterations_no_improve + 1}/{max_no_improve}")
            
            # ILS: Perturbação após estagnação (usando max_no_improve // 2)
            if iterations_no_improve > max_no_improve // 2:
                self.logger.debug("Estagnação detectada. Aplicando perturbação ILS.")
                self.current_matrix = self.perturb_solution()
                current_score = evaluation_func(self.current_matrix)
                self.perturbation_size = min(
                    self.max_perturbation,
                    self.perturbation_size + 1
                )
                self.logger.debug(
                    f"Perturbação aplicada (size={self.perturbation_size}). Novo score: {current_score:.4f}"
                )
            
            # VNS: Exploração de vizinhanças
            improved = False
            neighborhoods = [
                self._frequency_based_neighborhood,
                self._conservation_based_neighborhood,
                self._group_based_neighborhood
            ]
            
            for k, neighborhood in enumerate(neighborhoods):
                self.logger.debug(f"Explorando vizinhança {k}: {neighborhood.__doc__}")
                neighbor = neighborhood()
                neighbor_score = evaluation_func(neighbor)
                self.logger.debug(f"Score da vizinhança {k}: {neighbor_score:.4f}")
                
                if neighbor_score > current_score + self.min_improvement:
                    self.logger.debug(f"Melhoria encontrada na vizinhança {k}: {current_score:.4f} -> {neighbor_score:.4f}")
                    score_improvement = neighbor_score - current_score
                    self.current_matrix = neighbor
                    current_score = neighbor_score
                    self._store_improvement(score_improvement, k)
                    improved = True
                    break  # Sai do loop de vizinhanças
            
            # Atualiza melhor solução
            if current_score > self.best_score + self.min_improvement:
                self.best_matrix = self.current_matrix.copy()
                self.best_score = current_score
                iterations_no_improve = 0
                self.perturbation_size = 2  # Reset perturbação
                self.logger.info(f"Nova melhor pontuação encontrada: {self.best_score:.4f}")
            else:
                iterations_no_improve += 1
                self.logger.debug(f"Sem melhoria na iteração. Iterações sem melhoria: {iterations_no_improve}")
        
        # Restaura melhor solução encontrada
        self.current_matrix = self.best_matrix.copy()
        self.logger.info(f"Busca VNS-ILS concluída. Melhor score: {self.best_score:.6f}")
        return self.best_score

    def save_best_matrix(self, best_matrix: AdaptiveMatrix, results_dir: Path, run_info: str) -> None:
        """
        Salva a melhor matriz encontrada com informações da execução.
        
        Args:
            best_matrix (AdaptiveMatrix): A melhor matriz encontrada.
            results_dir (Path): Diretório onde a matriz será salva.
            run_info (str): Informações adicionais sobre a execução.
        """
        try:
            # Adicionar informações da execução
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            filename = f"{timestamp}-AdaptivePAM-{run_info}.txt"
            output_path = results_dir / filename
            
            best_matrix.to_clustalw_format(output_path)
            self.logger.info(f"Best matrix saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar a melhor matriz: {e}")
            raise
