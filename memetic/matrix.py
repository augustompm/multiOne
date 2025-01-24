# memetic/matrix.py

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Set, Optional
import json

class AdaptiveMatrix:
    """
    Matriz de substituição adaptativa com restrições biológicas.
    
    Gerencia uma matriz PAM adaptativa mantendo:
    - Restrições biológicas e físico-químicas
    - Simetria da matriz
    - Limites de scores por região
    - Rastreamento de mudanças
    """
    
    def __init__(self, hyperparams: Optional[Dict] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ordem padrão dos aminoácidos (20 padrão + especiais)
        self.aa_order = "ARNDCQEGHILKMFPSTWYV"
        self.aa_to_index = {aa: i for i, aa in enumerate(self.aa_order)}
        
        # Grupos físico-químicos
        self.similar_groups = [
            {'I', 'L', 'V', 'M'},         # Hidrofóbicos alifáticos
            {'F', 'Y', 'W'},              # Aromáticos
            {'K', 'R', 'H'},              # Básicos
            {'D', 'E'},                   # Ácidos
            {'S', 'T'},                   # Hidroxílicos
            {'N', 'Q'},                   # Amidas
            {'A', 'G'},                   # Pequenos
            {'P'},                        # Especial (prolina)
            {'C'}                         # Especial (cisteína)
        ]
        
        # Hyperparameters
        self.hyperparams = hyperparams or {
            'MATRIX': {
                'SCORE_DIAGONAL': {'min': -2, 'max': 17},
                'SCORE_SIMILAR': {'min': -4, 'max': 8},
                'SCORE_DIFFERENT': {'min': -8, 'max': 4},
                'MAX_ADJUSTMENT': 2
            }
        }
        
        # Inicializa matriz
        self.matrix = self._load_pam250()
        
        # Guarda cópia da original
        self.original_matrix = self.matrix.copy()
        
        # Rastreamento de mudanças
        self.changes_history = []
        
    def _load_pam250(self) -> np.ndarray:
        """Carrega matriz PAM250 com validações."""
        matrix = np.zeros((20, 20))
        
        try:
            current_dir = Path(__file__).parent
            pam_path = current_dir / "matrices" / "pam250.txt"
            
            with open(pam_path) as f:
                lines = f.readlines()
                
            # Filtra linhas de dados
            data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
            
            if not data_lines:
                raise ValueError("PAM250 file is empty or malformed")
                
            # Processa matrix
            for i, line in enumerate(data_lines[1:21]):  # Primeiros 20 AAs
                parts = line.strip().split()
                if len(parts) < 21:  # AA + 20 scores
                    raise ValueError(f"Invalid line format: {line}")
                    
                values = parts[1:21]  # Skip AA letter
                matrix[i] = [float(x) for x in values]
                
            # Valida simetria
            if not np.allclose(matrix, matrix.T):
                self.logger.warning("Loaded PAM250 matrix is not symmetric")
                # Força simetria
                matrix = (matrix + matrix.T) / 2
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error loading PAM250: {e}")
            raise
            
    def get_score(self, aa1: str, aa2: str) -> int:
        """Retorna score com validação."""
        try:
            i = self.aa_to_index[aa1]
            j = self.aa_to_index[aa2]
            return int(self.matrix[i][j])
        except KeyError:
            self.logger.warning(f"Invalid amino acids: {aa1}, {aa2}")
            return -8  # Penalidade padrão PAM
            
    def update_score(self, aa1: str, aa2: str, new_score: int) -> bool:
        """
        Atualiza score mantendo restrições biológicas.
        
        Args:
            aa1, aa2: Aminoácidos
            new_score: Novo score proposto
            
        Returns:
            bool: True se atualização foi aceita
        """
        try:
            i = self.aa_to_index[aa1]
            j = self.aa_to_index[aa2]
            
            # Valida limites baseado no tipo de relação
            if not self._validate_score(aa1, aa2, new_score):
                return False
                
            # Aplica atualização simétrica
            self.matrix[i][j] = new_score
            self.matrix[j][i] = new_score
            
            # Registra mudança
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
        """Validação completa de scores."""
        
        # Define limites baseado no tipo de relação
        if aa1 == aa2:  # Diagonal
            limits = self.hyperparams['MATRIX']['SCORE_DIAGONAL']
        elif self._are_similar(aa1, aa2):  # AAs similares
            limits = self.hyperparams['MATRIX']['SCORE_SIMILAR']
        else:  # AAs diferentes
            limits = self.hyperparams['MATRIX']['SCORE_DIFFERENT']
            
        # Valida limites absolutos
        if not limits['min'] <= new_score <= limits['max']:
            return False
            
        # Valida magnitude da mudança
        old_score = self.get_score(aa1, aa2)
        if abs(new_score - old_score) > self.hyperparams['MATRIX']['MAX_ADJUSTMENT']:
            return False
            
        # Valida relações físico-químicas
        if not self._validate_physicochemical(aa1, aa2, new_score):
            return False
            
        return True
        
    def _are_similar(self, aa1: str, aa2: str) -> bool:
        """Verifica se AAs pertencem ao mesmo grupo."""
        return any(aa1 in group and aa2 in group for group in self.similar_groups)
        
    def _validate_physicochemical(self, aa1: str, aa2: str, new_score: int) -> bool:
        """Valida relações físico-químicas."""
        
        # AAs do mesmo grupo devem ter scores maiores
        if self._are_similar(aa1, aa2):
            for other_aa in self.aa_order:
                if not self._are_similar(aa1, other_aa):
                    other_score = self.get_score(aa1, other_aa)
                    if new_score <= other_score:
                        return False
                        
        # Scores na diagonal devem ser os maiores
        if aa1 == aa2:
            for other_aa in self.aa_order:
                if other_aa != aa1:
                    other_score = self.get_score(aa1, other_aa)
                    if new_score <= other_score:
                        return False
                        
        return True
        
    def to_clustalw_format(self, output_path: Path) -> None:
        """Salva matriz no formato ClustalW com metadados."""
        try:
            # Ordem completa incluindo AAs especiais
            full_aa_order = list("ARNDCQEGHILKMFPSTWYVBZX*")
            
            with open(output_path, 'w') as f:
                # Cabeçalho
                f.write("#\n")
                f.write("#  Adaptive PAM Matrix\n")
                f.write("#  Generated with enhanced biological constraints\n")
                f.write("#\n")
                
                # Linha de aminoácidos
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
                    
            # Salva metadados
            meta_path = output_path.with_suffix('.meta.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'changes': self.changes_history,
                    'hyperparams': self.hyperparams
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving matrix: {e}")
            raise
            
    def copy(self) -> 'AdaptiveMatrix':
        """Cria cópia profunda."""
        new_matrix = AdaptiveMatrix(self.hyperparams)
        new_matrix.matrix = self.matrix.copy()
        new_matrix.original_matrix = self.original_matrix.copy()
        new_matrix.changes_history = self.changes_history.copy()
        return new_matrix
        
    def reset(self) -> None:
        """Reseta para PAM250 original."""
        self.matrix = self.original_matrix.copy()
        self.changes_history.clear()
        
    def get_changes_summary(self) -> Dict:
        """Retorna sumário das mudanças aplicadas."""
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
            ]),
            'most_changed_pairs': self._get_most_changed_pairs()
        }
        
    def _get_most_changed_pairs(self, top_n: int = 5) -> List[Tuple[str, str, int]]:
        """Retorna pares de AAs mais modificados."""
        pair_changes = defaultdict(int)
        
        for change in self.changes_history:
            pair = tuple(sorted([change['aa1'], change['aa2']]))
            pair_changes[pair] += abs(change['new_score'] - change['old_score'])
            
        return sorted(
            [(aa1, aa2, total) for (aa1, aa2), total in pair_changes.items()],
            key=lambda x: x[2],
            reverse=True
        )[:top_n]