# memetic/local_search.py

import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict
import logging
import random
from .matrix import AdaptiveMatrix

class LocalSearch:
    """
    Variable Neighborhood Search implementation for matrix optimization using
    three biologically-motivated neighborhood structures.
    """
    
    def __init__(self, matrix: AdaptiveMatrix):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Inicializando a classe LocalSearch.")
        
        self.matrix = matrix
        self.logger.debug("Matriz adaptativa inicializada.")
        
        self.substitution_frequencies = defaultdict(lambda: defaultdict(float))
        self.logger.debug("Frequências de substituição inicializadas.")
        
        self.conservation_weights = defaultdict(float)
        self.logger.debug("Pesos de conservação inicializados.")
        
        self.best_score = float('-inf')
        self.logger.debug(f"Melhor score inicializado para {self.best_score}.")
        
        # Amino acid groups based on physicochemical properties
        self.aa_groups = {
            'hydrophobic': {'I', 'L', 'V', 'M', 'F', 'W', 'A'},
            'polar': {'S', 'T', 'N', 'Q'},
            'acidic': {'D', 'E'},
            'basic': {'K', 'R', 'H'},
            'special': {'C', 'G', 'P', 'Y'}
        }
        self.logger.debug(f"Grupos de aminoácidos definidos: {self.aa_groups.keys()}.")
        
    def analyze_alignment(self, xml_path: Path) -> None:
        """
        Analyzes alignment to extract substitution frequencies and conservation weights.
        """
        self.logger.info(f"Iniciando análise de alinhamento no arquivo: {xml_path}.")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            self.logger.debug("Arquivo XML parseado com sucesso.")
        except ET.ParseError as pe:
            self.logger.error(f"Erro ao parsear o arquivo XML: {pe}")
            raise
        except FileNotFoundError as fnf:
            self.logger.error(f"Arquivo XML não encontrado: {fnf}")
            raise
        except Exception as e:
            self.logger.error(f"Erro ao carregar o arquivo XML: {e}")
            raise
        
        # Extract sequences and their blocks
        sequences = {}
        blocks = []
        self.logger.debug("Extraindo sequências e blocos do alinhamento.")
        
        for seq in root.findall(".//sequence"):
            seq_name = seq.find("seq-name").text
            seq_data = seq.find("seq-data").text.strip()
            sequences[seq_name] = seq_data
            self.logger.debug(f"Sequência extraída: {seq_name} com {len(seq_data)} caracteres.")
            
            # Get blocks and their conservation scores
            for block in seq.findall(".//fitem/[ftype='BLOCK']"):
                start = int(block.find("fstart").text)
                stop = int(block.find("fstop").text)
                score = float(block.find("fscore").text)
                
                block_data = {
                    'seq': seq_name,
                    'start': start,
                    'stop': stop,
                    'score': score,
                    'data': seq_data[start-1:stop]
                }
                blocks.append(block_data)
                self.logger.debug(f"Bloco extraído: {block_data}.")
                
        self.logger.info(f"Total de sequências extraídas: {len(sequences)}.")
        self.logger.info(f"Total de blocos extraídos: {len(blocks)}.")
        
        self._analyze_patterns(sequences, blocks)
        self.logger.info("Análise de padrões concluída.")
        
    def _analyze_patterns(self, sequences: Dict[str, str], blocks: List[Dict]) -> None:
        """
        Analyzes substitution patterns and conservation in blocks.
        """
        self.logger.info("Iniciando análise de padrões e conservação.")
        
        # Reset counters
        self.substitution_frequencies.clear()
        self.conservation_weights.clear()
        self.logger.debug("Frequências de substituição e pesos de conservação resetados.")
        
        # Group blocks by aligned positions
        aligned_blocks = defaultdict(list)
        for block in blocks:
            aligned_blocks[block['start']].append(block)
        self.logger.debug(f"Blocos agrupados por posição alinhada: {len(aligned_blocks)} grupos.")
        
        # Analyze each aligned block
        for pos, block_group in aligned_blocks.items():
            self.logger.debug(f"Analisando posição alinhada: {pos} com {len(block_group)} blocos.")
            
            # Calculate position conservation
            avg_score = np.mean([b['score'] for b in block_group])
            self.logger.debug(f"Score de conservação médio na posição {pos}: {avg_score}.")
            
            # Analyze substitutions
            for i, block1 in enumerate(block_group):
                for j, block2 in enumerate(block_group[i+1:], i+1):
                    for offset in range(min(len(block1['data']), len(block2['data']))):
                        aa1 = block1['data'][offset]
                        aa2 = block2['data'][offset]
                        
                        if aa1 != '-' and aa2 != '-':
                            # Weight by conservation
                            weight = avg_score
                            self.substitution_frequencies[aa1][aa2] += weight
                            self.substitution_frequencies[aa2][aa1] += weight
                            self.conservation_weights[aa1] += weight
                            self.conservation_weights[aa2] += weight
                            self.logger.debug(
                                f"Substituição: {aa1} <-> {aa2} com peso {weight}."
                            )
        
        self.logger.info("Análise de padrões e conservação finalizada.")
        
    def vns_search(self, evaluation_func, max_iterations: int = 100,
                  max_no_improve: int = 20) -> float:
        """
        Performs VNS using three neighborhood structures.
        """
        self.logger.info("Iniciando busca VNS.")
        
        current_score = evaluation_func(self.matrix)
        self.best_score = current_score
        self.logger.debug(f"Score inicial: {self.best_score}.")
        
        iterations_no_improve = 0
        
        neighborhoods = [
            self._frequency_based_neighborhood,
            self._conservation_based_neighborhood,
            self._group_based_neighborhood
        ]
        self.logger.debug(f"Estruturas de vizinhança definidas: {[n.__name__ for n in neighborhoods]}.")
        
        current_neighborhood = 0
        
        while (iterations_no_improve < max_no_improve and 
               current_neighborhood < len(neighborhoods)):
            
            self.logger.debug(
                f"Iteração VNS: Vizinhança atual {current_neighborhood + 1} de {len(neighborhoods)}."
            )
            
            # Generate neighbor using current neighborhood structure
            neighborhood_func = neighborhoods[current_neighborhood]
            neighbor_matrix = neighborhood_func()
            self.logger.debug(
                f"Vizinhança {current_neighborhood + 1}: Gerando matriz vizinha usando {neighborhood_func.__name__}."
            )
            
            # Evaluate neighbor
            neighbor_score = evaluation_func(neighbor_matrix)
            self.logger.debug(f"Score da matriz vizinha: {neighbor_score}.")
            
            if neighbor_score > self.best_score:
                # Accept improvement
                self.matrix = neighbor_matrix
                self.best_score = neighbor_score
                self.logger.info(
                    f"Melhoria encontrada: Novo melhor score {self.best_score}."
                )
                current_neighborhood = 0  # Reset to first neighborhood
                iterations_no_improve = 0
            else:
                # Try next neighborhood
                self.logger.debug(
                    f"Sem melhoria na vizinhança {current_neighborhood + 1}. Passando para a próxima."
                )
                current_neighborhood += 1
                iterations_no_improve += 1
            
            self.logger.debug(
                f"Estado atual - Melhor score: {self.best_score}, Iterações sem melhoria: {iterations_no_improve}."
            )
                
            self.logger.debug(
                f"VNS Iteration - Score: {neighbor_score}, Best: {self.best_score}, "
                f"Neighborhood: {current_neighborhood}"
            )
                
        self.logger.info(f"Busca VNS concluída com melhor score: {self.best_score}.")
        return self.best_score
            
    def _frequency_based_neighborhood(self) -> AdaptiveMatrix:
        """
        Generates neighbor by updating highly frequent substitutions.
        """
        self.logger.debug("Gerando vizinhança baseada em frequência de substituições.")
        
        new_matrix = AdaptiveMatrix()
        new_matrix.matrix = self.matrix.matrix.copy()
        self.logger.debug("Matriz vizinha criada a partir da matriz atual.")
        
        # Select substitutions based on frequency
        sorted_substitutions = [
            (aa1, aa2, freq)
            for aa1, subs in self.substitution_frequencies.items()
            for aa2, freq in subs.items()
        ]
        sorted_substitutions.sort(key=lambda x: x[2], reverse=True)
        self.logger.debug(
            f"Substituições ordenadas por frequência: {sorted_substitutions[:5]} (top 5)."
        )
        
        # Update top substitutions
        for aa1, aa2, freq in sorted_substitutions[:5]:  # Adjust number as needed
            current_score = new_matrix.get_score(aa1, aa2)
            # Adjust score based on frequency
            adjustment = int(np.log2(freq + 1))
            new_score = max(-8, min(17, current_score + adjustment))  # PAM250 bounds
            success = new_matrix.update_score(aa1, aa2, new_score)
            if success:
                self.logger.debug(
                    f"Atualizando score de {aa1} <-> {aa2} de {current_score} para {new_score}."
                )
            else:
                self.logger.warning(
                    f"Falha ao atualizar score de {aa1} <-> {aa2} para {new_score}."
                )
                
        self.logger.debug("Vizinhança baseada em frequência de substituições gerada.")
        return new_matrix
            
    def _conservation_based_neighborhood(self) -> AdaptiveMatrix:
        """
        Generates neighbor by focusing on highly conserved positions.
        """
        self.logger.debug("Gerando vizinhança baseada em conservação.")
        
        new_matrix = AdaptiveMatrix()
        new_matrix.matrix = self.matrix.matrix.copy()
        self.logger.debug("Matriz vizinha criada a partir da matriz atual.")
        
        # Select amino acids based on conservation weight
        conserved_aas = sorted(
            self.conservation_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 most conserved
        self.logger.debug(
            f"Aminoácidos mais conservados selecionados: {conserved_aas}."
        )
        
        for aa, weight in conserved_aas:
            # Adjust scores for this conserved amino acid
            for other_aa in self.matrix.aa_order:
                if other_aa != aa:
                    current_score = new_matrix.get_score(aa, other_aa)
                    # More conservative adjustments for highly conserved positions
                    adjustment = int(weight / max(self.conservation_weights.values()) * 2)
                    new_score = max(-8, min(17, current_score + adjustment))
                    success = new_matrix.update_score(aa, other_aa, new_score)
                    if success:
                        self.logger.debug(
                            f"Atualizando score de {aa} <-> {other_aa} de {current_score} para {new_score}."
                        )
                    else:
                        self.logger.warning(
                            f"Falha ao atualizar score de {aa} <-> {other_aa} para {new_score}."
                        )
                    
        self.logger.debug("Vizinhança baseada em conservação gerada.")
        return new_matrix
            
    def _group_based_neighborhood(self) -> AdaptiveMatrix:
        """
        Generates neighbor by updating physicochemically related groups.
        """
        self.logger.debug("Gerando vizinhança baseada em grupos físico-químicos.")
        
        new_matrix = AdaptiveMatrix()
        new_matrix.matrix = self.matrix.matrix.copy()
        self.logger.debug("Matriz vizinha criada a partir da matriz atual.")
        
        # Select a random group
        group_name = random.choice(list(self.aa_groups.keys()))
        group_aas = self.aa_groups[group_name]
        self.logger.debug(f"Grupo selecionado para atualização: {group_name} com aminoácidos {group_aas}.")
        
        # Adjust scores within group
        for aa1 in group_aas:
            for aa2 in group_aas:
                if aa1 != aa2:
                    current_score = new_matrix.get_score(aa1, aa2)
                    # Favor substitutions within same physicochemical group
                    new_score = min(17, current_score + 1)
                    success = new_matrix.update_score(aa1, aa2, new_score)
                    if success:
                        self.logger.debug(
                            f"Atualizando score de {aa1} <-> {aa2} de {current_score} para {new_score}."
                        )
                    else:
                        self.logger.warning(
                            f"Falha ao atualizar score de {aa1} <-> {aa2} para {new_score}."
                        )
                        
        self.logger.debug("Vizinhança baseada em grupos físico-químicos gerada.")
        return new_matrix
