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
        
        # Hyperparameters específicos para cada nível
        self.hyperparams = hyperparams or {
            'SCORE_DIAGONAL': {'min': -2, 'max': 17},
            'SCORE_SIMILAR': {'min': -4, 'max': 8},
            'SCORE_DIFFERENT': {'min': -8, 'max': 4},
            'MAX_ADJUSTMENT': 2
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
                limits = self.hyperparams['SCORE_DIAGONAL']
            elif any(aa1 in group and aa2 in group for group in self.similar_groups):
                limits = self.hyperparams['SCORE_SIMILAR']
            else:  # Diferentes
                limits = self.hyperparams['SCORE_DIFFERENT']
            
            # Valida limites absolutos
            if not limits['min'] <= new_score <= limits['max']:
                return False
            
            # Valida magnitude da mudança
            old_score = self.get_score(aa1, aa2)
            max_adjustment = self.hyperparams['MAX_ADJUSTMENT']
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

    def _write_clustalw_format(self, f) -> None:
        """Escreve matriz no formato ClustalW"""
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
                self._write_clustalw_format(f)
                
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
    def __init__(self, hyperparams: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hyperparams = hyperparams
        self.conservation_level = None

        # Matrizes para cada nível de conservação
        self.matrices = {
            'HIGH': AdaptiveMatrix(hyperparams['MATRIX']['HIGH']),
            'MEDIUM': AdaptiveMatrix(hyperparams['MATRIX']['MEDIUM']),
            'LOW': AdaptiveMatrix(hyperparams['MATRIX']['LOW'])
        }
        
        # Estrutura unificada para tracking de uso
        self.usage_stats = {
            level: {
                'usage_count': 0,
                'best_score': float('-inf'),
                'changes': 0
            }
            for level in ['HIGH', 'MEDIUM', 'LOW']
        }

    def set_conservation_level(self, level: str) -> None:
        """Define o nível de conservação atual"""
        if level in self.matrices:
            self.conservation_level = level
            self.logger.debug(f"Conservation level set to {level}")
        else:
            self.logger.error(f"Invalid conservation level: {level}")

    def get_matrix(self, conservation_level: str) -> Optional[AdaptiveMatrix]:
        """Retorna matriz para nível de conservação específico"""
        # Validação mais robusta
        if not conservation_level:
            self.logger.error("Conservation level must be specified")
            return None
            
        if conservation_level not in ['HIGH', 'MEDIUM', 'LOW']:
            self.logger.error(f"Invalid conservation level: {conservation_level}")
            return None
            
        matrix = self.matrices.get(conservation_level)
        if matrix:
            self.usage_stats[conservation_level]['usage_count'] += 1
        return matrix

    def _get_high_params(self, base_params: Dict) -> Dict:
        """Parâmetros para conservação alta baseados no BLOSUM80"""
        return base_params['MATRIX']['HIGH']

    def _get_medium_params(self, base_params: Dict) -> Dict:
        """Parâmetros para conservação média baseados no PAM120"""
        return base_params['MATRIX']['MEDIUM']

    def _get_low_params(self, base_params: Dict) -> Dict:
        """Parâmetros para conservação baixa baseados no PAM250"""
        return base_params['MATRIX']['LOW']

    def export_matrices(self, output_dir: Path) -> Dict[str, Path]:
        """Exporta matrizes no formato ClustalW"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        paths = {}
        
        # Exporta matriz combinada
        combined_path = output_dir / f"BioFit_combined_{timestamp}.mat"
        with open(combined_path, 'w') as f:
            f.write("# BioFit Combined Matrix\n")
            for level, matrix in self.matrices.items():
                f.write(f"\n# {level} Conservation Matrix\n")
                matrix._write_clustalw_format(f)
        paths['combined'] = combined_path
            
        # Exporta matrizes individuais
        for level, matrix in self.matrices.items():
            matrix_path = output_dir / f"BioFit_{level}_{timestamp}.mat"
            with open(matrix_path, 'w') as f:
                f.write(f"# {level} conservation substitution matrix\n")
                matrix._write_clustalw_format(f)
            paths[level] = matrix_path
        
        return paths

    def export_final_matrices(self, output_dir: Path, instance: str, execution_id: int, score: float, execution_time: float) -> Dict[str, Path]:
        """
        Versão completa para exportar os resultados finais.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        paths = {}
        
        # Matriz combinada
        combined_path = output_dir / f"{instance},{execution_id},{score:.4f},{execution_time:.1f}.mat"
        with open(combined_path, 'w') as f:
            f.write("# BioFit Combined Matrix\n")
            for level, matrix in self.matrices.items():
                f.write(f"\n# {level} Conservation Matrix\n")
                matrix._write_clustalw_format(f)
        paths['combined'] = combined_path
        
        # Matrizes individuais
        for level, matrix in self.matrices.items():
            matrix_path = output_dir / f"BioFit_{level}_{instance}_{timestamp}.mat"
            with open(matrix_path, 'w') as f:
                f.write(f"# {level} conservation substitution matrix\n")
                matrix._write_clustalw_format(f)
            paths[level] = matrix_path
        
        return paths

    def copy(self) -> 'MatrixManager':
        """Cria cópia profunda"""
        new_manager = MatrixManager(self.hyperparams)
        # Copia cada matriz individualmente
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            new_manager.matrices[level] = self.matrices[level].copy()
            new_manager.usage_stats[level] = self.usage_stats[level].copy()
        new_manager.conservation_level = self.conservation_level
        return new_manager

    def get_stats(self) -> Dict:
        """Retorna estatísticas atualizadas"""
        return {
            level: {
                'usage': self.usage_stats[level]['usage_count'],
                'best_score': self.usage_stats[level]['best_score'],
                'changes': len(self.matrices[level].changes_history)
            }
            for level in self.matrices.keys()
        }