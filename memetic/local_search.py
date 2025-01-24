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
        self.matrix = matrix
        self.substitution_frequencies = defaultdict(lambda: defaultdict(float))
        self.conservation_weights = defaultdict(float)
        self.best_score = float('-inf')
        
        # Amino acid groups based on physicochemical properties
        self.aa_groups = {
            'hydrophobic': {'I', 'L', 'V', 'M', 'F', 'W', 'A'},
            'polar': {'S', 'T', 'N', 'Q'},
            'acidic': {'D', 'E'},
            'basic': {'K', 'R', 'H'},
            'special': {'C', 'G', 'P', 'Y'}
        }
        
    def analyze_alignment(self, xml_path: Path) -> None:
        """
        Analyzes alignment to extract substitution frequencies and conservation weights.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract sequences and their blocks
        sequences = {}
        blocks = []
        
        for seq in root.findall(".//sequence"):
            seq_name = seq.find("seq-name").text
            seq_data = seq.find("seq-data").text.strip()
            sequences[seq_name] = seq_data
            
            # Get blocks and their conservation scores
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
                
        self._analyze_patterns(sequences, blocks)
        
    def _analyze_patterns(self, sequences: Dict[str, str], blocks: List[Dict]) -> None:
        """
        Analyzes substitution patterns and conservation in blocks.
        """
        # Reset counters
        self.substitution_frequencies.clear()
        self.conservation_weights.clear()
        
        # Group blocks by aligned positions
        aligned_blocks = defaultdict(list)
        for block in blocks:
            aligned_blocks[block['start']].append(block)
            
        # Analyze each aligned block
        for pos, block_group in aligned_blocks.items():
            # Calculate position conservation
            avg_score = np.mean([b['score'] for b in block_group])
            
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
                            
    def vns_search(self, evaluation_func, max_iterations: int = 100,
                  max_no_improve: int = 20) -> float:
        """
        Performs VNS using three neighborhood structures.
        """
        current_score = evaluation_func(self.matrix)
        self.best_score = current_score
        iterations_no_improve = 0
        
        neighborhoods = [
            self._frequency_based_neighborhood,
            self._conservation_based_neighborhood,
            self._group_based_neighborhood
        ]
        
        current_neighborhood = 0
        
        while (iterations_no_improve < max_no_improve and 
               current_neighborhood < len(neighborhoods)):
            
            # Generate neighbor using current neighborhood structure
            neighbor_matrix = neighborhoods[current_neighborhood]()
            
            # Evaluate neighbor
            neighbor_score = evaluation_func(neighbor_matrix)
            
            if neighbor_score > self.best_score:
                # Accept improvement
                self.matrix = neighbor_matrix
                self.best_score = neighbor_score
                current_neighborhood = 0  # Reset to first neighborhood
                iterations_no_improve = 0
            else:
                # Try next neighborhood
                current_neighborhood += 1
                iterations_no_improve += 1
                
            logging.debug(f"VNS Iteration - Score: {neighbor_score}, "
                        f"Best: {self.best_score}, "
                        f"Neighborhood: {current_neighborhood}")
                
        return self.best_score
        
    def _frequency_based_neighborhood(self) -> AdaptiveMatrix:
        """
        Generates neighbor by updating highly frequent substitutions.
        """
        new_matrix = AdaptiveMatrix()
        new_matrix.matrix = self.matrix.matrix.copy()
        
        # Select substitutions based on frequency
        sorted_substitutions = [
            (aa1, aa2, freq)
            for aa1, subs in self.substitution_frequencies.items()
            for aa2, freq in subs.items()
        ]
        sorted_substitutions.sort(key=lambda x: x[2], reverse=True)
        
        # Update top substitutions
        for aa1, aa2, freq in sorted_substitutions[:5]:  # Adjust number as needed
            current_score = new_matrix.get_score(aa1, aa2)
            # Adjust score based on frequency
            adjustment = int(np.log2(freq + 1))
            new_score = max(-8, min(17, current_score + adjustment))  # PAM250 bounds
            new_matrix.update_score(aa1, aa2, new_score)
            
        return new_matrix
        
    def _conservation_based_neighborhood(self) -> AdaptiveMatrix:
        """
        Generates neighbor by focusing on highly conserved positions.
        """
        new_matrix = AdaptiveMatrix()
        new_matrix.matrix = self.matrix.matrix.copy()
        
        # Select amino acids based on conservation weight
        conserved_aas = sorted(
            self.conservation_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 most conserved
        
        for aa, weight in conserved_aas:
            # Adjust scores for this conserved amino acid
            for other_aa in self.matrix.aa_order:
                if other_aa != aa:
                    current_score = new_matrix.get_score(aa, other_aa)
                    # More conservative adjustments for highly conserved positions
                    adjustment = int(weight / max(self.conservation_weights.values()) * 2)
                    new_score = max(-8, min(17, current_score + adjustment))
                    new_matrix.update_score(aa, other_aa, new_score)
                    
        return new_matrix
        
    def _group_based_neighborhood(self) -> AdaptiveMatrix:
        """
        Generates neighbor by updating physicochemically related groups.
        """
        new_matrix = AdaptiveMatrix()
        new_matrix.matrix = self.matrix.matrix.copy()
        
        # Select a random group
        group_name = random.choice(list(self.aa_groups.keys()))
        group_aas = self.aa_groups[group_name]
        
        # Adjust scores within group
        for aa1 in group_aas:
            for aa2 in group_aas:
                if aa1 != aa2:
                    current_score = new_matrix.get_score(aa1, aa2)
                    # Favor substitutions within same physicochemical group
                    new_score = min(17, current_score + 1)
                    new_matrix.update_score(aa1, aa2, new_score)
                    
        return new_matrix