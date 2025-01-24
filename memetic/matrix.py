# memetic/matrix.py

import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set
import logging

class AdaptiveMatrix:
    """
    A class representing an adaptive substitution matrix that can be modified
    based on alignment patterns while maintaining biological constraints.
    """
    
    # Variável de classe para controlar o warning
    _order_warning_shown = False
    
    def __init__(self):
        # Inicializa o logger para a instância
        self.logger = logging.getLogger(__name__)
        
        # Ordem padrão dos aminoácidos - incluindo os extras para combinar com PAM250
        self.aa_order = "ARNDCQEGHILKMFPSTWYV"
        self.aa_to_index = {aa: i for i, aa in enumerate(self.aa_order)}
        
        # Grupos físico-químicos dos aminoácidos
        self.similar_groups = [
            {'I', 'L', 'V', 'M'},  # Hidrofóbicos
            {'R', 'K', 'H'},       # Básicos
            {'D', 'E'},            # Ácidos
            {'F', 'Y', 'W'},       # Aromáticos
            {'S', 'T'},            # Hidroxílicos
            {'N', 'Q'}             # Amídicos
        ]
        
        # Limite máximo de ajuste em relação à PAM250 original
        self.max_adjustment = 2
        
        # Ajustes diferentes para diferentes tipos de mudanças
        self.max_diagonal_adjustment = 3    # Para diagonal principal
        self.max_similar_adjustment = 2     # Para aminoácidos similares
        self.max_different_adjustment = 1   # Para outros casos
        
        # Carrega matriz PAM250
        self.matrix = self._load_pam250()
        
        # Guarda cópia da matriz original para referência
        self.original_matrix = self.matrix.copy()
        
    def _load_pam250(self) -> np.ndarray:
        """
        Loads the PAM250 matrix from the matrices directory.
        The PAM250 matrix file has a specific format with:
        - Comment lines starting with #
        - A header row with amino acid letters
        - Data rows starting with amino acid letter followed by scores
        """
        matrix = np.zeros((20, 20))
        
        # Get path relative to this file's location
        current_file_dir = Path(__file__).parent
        pam_path = current_file_dir / "matrices" / "pam250.txt"
        
        try:
            with open(pam_path) as f:
                lines = f.readlines()
                
            # Skip comment lines and find the header line
            data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
            
            if not data_lines:
                raise ValueError("PAM250 file is empty or only contains comments.")
            
            # First line contains the amino acid order
            header = data_lines[0].strip().split()
            
            # Log warning about order difference only once for the class
            if not AdaptiveMatrix._order_warning_shown:
                expected_order = ''.join(self.aa_order)
                found_order = ''.join(header[:20])
                if found_order != expected_order:
                    self.logger.warning(f"PAM250 matrix amino acid order differs from expected order")
                    self.logger.warning(f"Expected: {expected_order}")
                    self.logger.warning(f"Found: {found_order}")
                    AdaptiveMatrix._order_warning_shown = True
            
            # Process matrix values
            for i, line in enumerate(data_lines[1:21]):  # Process only 20 amino acids
                parts = line.strip().split()
                if len(parts) < 21:
                    raise ValueError(f"Line {i+2} does not contain enough values")
                
                aa = parts[0]
                values = parts[1:21]  # Skip the AA letter
                if aa != self.aa_order[i]:
                    self.logger.warning(f"Row {i+1} amino acid '{aa}' does not match expected '{self.aa_order[i]}'")
                
                try:
                    matrix[i] = [float(x) for x in values]  # Convert to float first
                except ValueError as ve:
                    self.logger.error(f"Invalid score value on line {i+2}: {ve}")
                    raise
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error loading PAM250 matrix: {e}")
            raise
                
    def get_score(self, aa1: str, aa2: str) -> int:
        """Returns the substitution score for two amino acids."""
        try:
            i = self.aa_to_index[aa1]
            j = self.aa_to_index[aa2]
            return self.matrix[i][j]
        except KeyError:
            return -8  # Standard PAM250 penalty for unknown amino acids
                
    def update_score(self, aa1: str, aa2: str, new_score: int) -> bool:
        """
        Updates the substitution score while maintaining constraints:
        - Matrix symmetry
        - Score bounds relative to original PAM250
        - Physicochemical relationships
        """
        try:
            i = self.aa_to_index[aa1]
            j = self.aa_to_index[aa2]
            
            # Escolhe o limite apropriado
            if i == j:  # Diagonal
                max_allowed = self.max_diagonal_adjustment
            elif self._are_similar(aa1, aa2):  # Aminoácidos similares
                max_allowed = self.max_similar_adjustment
            else:
                max_allowed = self.max_different_adjustment
                
            original_score = self.original_matrix[i][j]
            if abs(new_score - original_score) > max_allowed:
                self.logger.debug(f"Ajuste {aa1}-{aa2}: {new_score} excede limite {max_allowed}")
                return False
                
            # Maintain relative scoring relationships
            if not self._validate_physicochemical_constraints(aa1, aa2, new_score):
                return False
                
            # Update symmetrically
            self.matrix[i][j] = new_score
            self.matrix[j][i] = new_score
            
            return True
                
        except KeyError:
            return False
                
    def _are_similar(self, aa1: str, aa2: str) -> bool:
        """Verifica se dois aminoácidos estão no mesmo grupo físico-químico"""
        return any(aa1 in group and aa2 in group for group in self.similar_groups)
        
    def _validate_physicochemical_constraints(self, aa1: str, aa2: str, new_score: int) -> bool:
        """
        Ensures that the new score maintains proper physicochemical relationships
        between amino acid groups.
        """
        # Find which groups the amino acids belong to
        aa1_groups = {i for i, group in enumerate(self.similar_groups) if aa1 in group}
        aa2_groups = {i for i, group in enumerate(self.similar_groups) if aa2 in group}
        
        # If amino acids are in the same group, score should be higher than
        # scores with amino acids from different groups
        if aa1_groups & aa2_groups:
            group_index = list(aa1_groups & aa2_groups)[0]
            for other_aa in self.aa_order:
                if other_aa not in self.similar_groups[group_index]:
                    if new_score <= self.get_score(aa1, other_aa):
                        return False
                        
        return True
                
    def to_clustalw_format(self, output_path: Path):
        """
        Salva matriz no formato ClustalW.
        Inclui cabeçalho com comentários e todos os aminoácidos extras.
        """
        try:
            # Ordem completa dos aminoácidos incluindo especiais
            full_aa_order = list("ARNDCQEGHILKMFPSTWYVBZX*")
            
            with open(output_path, 'w') as f:
                # Escreve cabeçalho com comentários
                f.write("#\n")
                f.write("#  Adaptive PAM Matrix\n")
                f.write("#\n")
                f.write("#  Based on PAM250\n")
                f.write("#\n")
                
                # Escreve linha de aminoácidos com espaçamento correto
                f.write("    " + "    ".join(full_aa_order) + "\n")
                
                # Escreve cada linha da matriz
                for aa1 in full_aa_order:
                    line = [aa1]
                    
                    for aa2 in full_aa_order:
                        # Casos especiais
                        if aa1 == '*' and aa2 == '*':
                            score = 1
                        elif aa1 == '*' or aa2 == '*':
                            score = -8
                        elif aa1 in ['B','Z','X'] or aa2 in ['B','Z','X']:
                            score = 0
                        else:
                            score = int(round(self.get_score(aa1, aa2)))
                            
                        # Formata com 4 espaços totais
                        line.append(f"{score:4d}")
                        
                    f.write(" ".join(line) + "\n")
                    
        except Exception as e:
            self.logger.error(f"Erro salvando matriz ClustalW: {e}")
            raise
