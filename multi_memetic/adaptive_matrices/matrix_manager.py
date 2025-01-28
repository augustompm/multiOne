# multi_memetic/adaptive_matrices/matrix_manager.py

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Set, Optional
import json
import time
from datetime import datetime
from collections import defaultdict

class AdaptiveMatrix:
    """
    Matriz de substituição adaptativa mantendo estrutura original do matrix.py,
    com restrições biológicas e físico-químicas do ClustalW.
    """
    def __init__(self, hyperparams: Optional[Dict] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ordem padrão dos aminoácidos (20 padrão + especiais)
        self.aa_order = "ARNDCQEGHILKMFPSTWYV"
        self.aa_to_index = {aa: i for i, aa in enumerate(self.aa_order)}
        
        # Hyperparameters - mantém estrutura original
        self.hyperparams = hyperparams or {
            'MATRIX': {
                'SCORE_DIAGONAL': {'min': -2, 'max': 17},
                'SCORE_SIMILAR': {'min': -4, 'max': 8},
                'SCORE_DIFFERENT': {'min': -8, 'max': 4},
                'MAX_ADJUSTMENT': 2
            }
        }
        
        # Define similar amino acid groups
        self.similar_groups = [
            ['A', 'G', 'V'],
            ['L', 'I', 'M'],
            ['F', 'Y', 'W'],
            # Add more groups as needed
        ]
        
        # Inicializa matriz
        self.matrix = self._load_pam250()
        self.original_matrix = self.matrix.copy()
        self.changes_history = []

    def get_score(self, aa1: str, aa2: str) -> int:
        """Retorna score entre dois aminoácidos"""
        try:
            aa1, aa2 = aa1.upper(), aa2.upper()
            
            if aa1 not in self.aa_order or aa2 not in self.aa_order:
                self.logger.debug(f"Invalid amino acids: {aa1}, {aa2}")
                return -8
                
            i = self.aa_to_index[aa1]
            j = self.aa_to_index[aa2]
            return int(self.matrix[i][j])
        except KeyError:
            return -8

    def update_score(self, aa1: str, aa2: str, new_score: int) -> bool:
        """Atualiza score mantendo restrições biológicas"""
        try:
            i = self.aa_to_index[aa1]
            j = self.aa_to_index[aa2]
            
            if not self._validate_score(aa1, aa2, new_score):
                return False
                
            self.matrix[i][j] = new_score
            self.matrix[j][i] = new_score
            
            self.changes_history.append({
                'aa1': aa1,
                'aa2': aa2,
                'old_score': self.matrix[i][j],
                'new_score': new_score
            })
            
            return True
            
        except KeyError:
            return False

    def _validate_score(self, aa1: str, aa2: str, new_score: int) -> bool:
        """Validação completa de scores"""
        try:
            # Define limites baseado no tipo de relação
            if aa1 == aa2:  # Diagonal
                limits = self.hyperparams['MATRIX'].get('SCORE_DIAGONAL', {'min': -2, 'max': 17})
            elif any(aa1 in group and aa2 in group for group in self.similar_groups):
                limits = self.hyperparams['MATRIX'].get('SCORE_SIMILAR', {'min': -4, 'max': 8})
            else:  # Diferentes
                limits = self.hyperparams['MATRIX'].get('SCORE_DIFFERENT', {'min': -8, 'max': 4})
            
            # Valida limites absolutos
            if not limits['min'] <= new_score <= limits['max']:
                return False
            
            # Valida magnitude da mudança
            old_score = self.get_score(aa1, aa2)
            # Aqui estava o erro - MAX_ADJUSTMENT estava no mesmo nível que SCORE_DIAGONAL
            max_adjustment = self.hyperparams['MATRIX'].get('MAX_ADJUSTMENT', 2)
            if abs(new_score - old_score) > max_adjustment:
                return False
            
            return True
            
        except KeyError as e:
            self.logger.error(f"Invalid hyperparameter access: {e}")
            return False

    def _load_pam250(self) -> np.ndarray:
        """Carrega matriz PAM250 com validações"""
        matrix = np.zeros((20, 20))
        
        try:
            # Ajusta para usar o path correto
            current_dir = Path(__file__).parent.parent  # Sobe para multi_memetic
            pam_path = current_dir / "matrices/pam250.txt"
            
            self.logger.debug(f"Loading PAM250 from: {pam_path}")
            
            with open(pam_path) as f:
                lines = f.readlines()
                
            data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
            
            if not data_lines:
                raise ValueError("PAM250 file is empty or malformed")
                
            for i, line in enumerate(data_lines[1:21]):
                parts = line.strip().split()
                if len(parts) < 21:
                    raise ValueError(f"Invalid line format: {line}")
                    
                values = parts[1:21]
                matrix[i] = [float(x) for x in values]
                
            if not np.allclose(matrix, matrix.T):
                self.logger.warning("Loaded PAM250 matrix is not symmetric")
                matrix = (matrix + matrix.T) / 2
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error loading PAM250: {e}")
            raise

    def _write_to_file(self, f) -> None:
        """Escreve matriz no arquivo já aberto"""
        try:
            # Ordem completa incluindo AAs especiais
            full_aa_order = list("ARNDCQEGHILKMFPSTWYVBZX*")
            
            # Escreve linha de aminoácidos
            f.write("    " + "    ".join(full_aa_order) + "\n")
            
            # Matriz
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
                        score = self.get_score(aa1, aa2)
                    line.append(f"{score:4d}")
                f.write(" ".join(line) + "\n")
                
        except Exception as e:
            self.logger.error(f"Error writing matrix to file: {e}")
            raise

    def to_clustalw_format(self, output_path: Path) -> None:
        """Salva matriz no formato ClustalW com metadados"""
        try:
            with open(output_path, 'w') as f:
                self._write_to_file(f)
                
        except Exception as e:
            self.logger.error(f"Error saving matrix: {e}")
            raise

    def copy(self) -> 'AdaptiveMatrix':
        """Cria cópia profunda"""
        new_matrix = AdaptiveMatrix(self.hyperparams)
        new_matrix.matrix = self.matrix.copy()
        new_matrix.original_matrix = self.original_matrix.copy()
        new_matrix.changes_history = self.changes_history.copy()
        return new_matrix

    def reset(self) -> None:
        """Reseta para PAM250 original"""
        self.matrix = self.original_matrix.copy()
        self.changes_history.clear()

    def get_changes_summary(self) -> Dict:
        """Retorna sumário das mudanças aplicadas"""
        if not self.changes_history:
            return {}
            
        return {
            'total_changes': len(self.changes_history),
            'avg_adjustment': np.mean([
                abs(c['new_score'] - c['old_score'])
                for c in self.changes_history
            ]),
            'max_adjustment': max([
                abs(c['new_score'] - c['old_score'])
                for c in self.changes_history
            ])
        }

class MatrixManager:
    """
    Gerencia conjunto de três matrizes adaptativas com base no ClustalW.
    Mantém estrutura de dados e validações do código original.
    """
    def __init__(self, hyperparams: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Matrizes para cada nível de conservação
        self.matrices = {
            'HIGH': AdaptiveMatrix(self._get_high_params(hyperparams)),
            'MEDIUM': AdaptiveMatrix(self._get_medium_params(hyperparams)),
            'LOW': AdaptiveMatrix(self._get_low_params(hyperparams))
        }
        
        self.usage_count = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        self.changes_history = []
        self.hyperparams = hyperparams

    def _get_high_params(self, base_params: Dict) -> Dict:
        """Parâmetros para conservação alta baseados no BLOSUM80"""
        params = base_params.copy()
        params['MATRIX'].update({
            'SCORE_DIAGONAL': {'min': -2, 'max': 17},
            'SCORE_SIMILAR': {'min': -4, 'max': 8},
            'SCORE_DIFFERENT': {'min': -8, 'max': 4},
        })
        return params

    def _get_medium_params(self, base_params: Dict) -> Dict:
        """Parâmetros para conservação média baseados no PAM120"""
        params = base_params.copy()
        params['MATRIX'].update({
            'SCORE_DIAGONAL': {'min': -2, 'max': 15},
            'SCORE_SIMILAR': {'min': -3, 'max': 7},
            'SCORE_DIFFERENT': {'min': -6, 'max': 3},
        })
        return params

    def _get_low_params(self, base_params: Dict) -> Dict:
        """Parâmetros para conservação baixa baseados no PAM250"""
        params = base_params.copy()
        params['MATRIX'].update({
            'SCORE_DIAGONAL': {'min': -2, 'max': 13},
            'SCORE_SIMILAR': {'min': -2, 'max': 6},
            'SCORE_DIFFERENT': {'min': -4, 'max': 3},
        })
        return params

    def get_matrix(self, conservation_level: str) -> AdaptiveMatrix:
        """Retorna matriz para nível de conservação específico"""
        matrix = self.matrices.get(conservation_level)
        if matrix:
            self.usage_count[conservation_level] += 1
        return matrix

    def export_matrices(self, output_dir: Path) -> Dict[str, Path]:
        """Exporta matrizes individuais e combinada"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        combined_path = output_dir / "temp_matrix.mat"
        with open(combined_path, 'w') as f:
            f.write("# Combined BioFit Matrix\n")
            for level, matrix in self.matrices.items():
                f.write(f"\n# {level} Conservation Matrix\n")
                matrix._write_to_file(f)  # Novo método para escrever no arquivo
            
        return {'combined': combined_path}

    def copy(self) -> 'MatrixManager':
        """Cria cópia profunda do gerenciador"""
        new_manager = MatrixManager(self.hyperparams)
        for level, matrix in self.matrices.items():
            new_manager.matrices[level] = matrix.copy()
        new_manager.usage_count = self.usage_count.copy()
        new_manager.changes_history = self.changes_history.copy()
        return new_manager

    def get_stats(self) -> Dict:
        """Retorna estatísticas do uso das matrizes"""
        return {
            'usage_count': self.usage_count,
            'changes': {
                level: matrix.get_changes_summary()
                for level, matrix in self.matrices.items()
            }
        }