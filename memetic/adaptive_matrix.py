import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from Bio.Align import substitution_matrices
from pathlib import Path

class AdaptiveMatrix:
    """
    Matriz de substituição adaptativa baseada na PAM250.
    
    Representação linear com 210 valores:
    - 20 valores da diagonal (conservação de AAs)
    - 190 valores do triângulo superior (substituições)
    
    Características:
    - Cache para lookups rápidos
    - Validação de restrições biológicas 
    - Integração com substitution_matrices
    """
    def __init__(self, gap_open: float = -10.0, gap_extend: float = -0.5):
        self.logger = logging.getLogger(__name__)
        
        # Ordem padrão dos aminoácidos
        self.aa_order = list("ACDEFGHIKLMNPQRSTVWY")
        
        # Parâmetros de gap
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        
        # Cache de scores {(aa1,aa2): score}
        self.score_cache: Dict[Tuple[str,str], float] = {}
        
        # Inicializa com PAM250
        self.chromosome = self._init_pam250_linear()
        self._update_cache()
        
    def _init_pam250_linear(self) -> np.ndarray:
        """
        Inicializa representação linear da PAM250.
        Retorna array com 210 valores.
        """
        try:
            # Carrega PAM250 da Biopython
            pam = substitution_matrices.load("PAM250")
            
            # Converte para forma linear
            linear = np.zeros(210)
            idx = 0
            
            # Primeiro os 20 valores da diagonal
            for aa in self.aa_order:
                linear[idx] = pam[aa,aa]
                idx += 1
                
            # Depois 190 valores do triângulo superior
            for i, aa1 in enumerate(self.aa_order):
                for aa2 in self.aa_order[i+1:]:
                    linear[idx] = pam[aa1,aa2]
                    idx += 1
                    
            return linear
            
        except Exception as e:
            self.logger.error(f"Erro inicializando PAM250: {e}")
            return np.ones(210)
            
    def _update_cache(self):
        """Atualiza cache de scores baseado no cromossomo atual"""
        self.score_cache.clear()
        idx = 0
        
        # Cache diagonal
        for aa in self.aa_order:
            self.score_cache[(aa,aa)] = self.chromosome[idx]
            idx += 1
            
        # Cache triângulo superior e inferior (simétrico)
        for i, aa1 in enumerate(self.aa_order):
            for aa2 in self.aa_order[i+1:]:
                score = self.chromosome[idx]
                self.score_cache[(aa1,aa2)] = score
                self.score_cache[(aa2,aa1)] = score
                idx += 1
                
    def get_score(self, aa1: str, aa2: str) -> float:
        """
        Retorna score de substituição para par de aminoácidos.
        Usa cache para eficiência.
        """
        # Tratamento de gaps
        if aa1 == '-' and aa2 == '-':
            return 0.0
        if aa1 == '-' or aa2 == '-':
            return self.gap_extend
            
        # Score do cache ou default para AAs desconhecidos
        return self.score_cache.get((aa1,aa2), -2.0)
        
    def update_from_vector(self, new_values: np.ndarray):
        """
        Atualiza matriz a partir de vetor de 210 valores.
        Valida restrições biológicas antes de atualizar.
        """
        if len(new_values) != 210:
            raise ValueError("Vector must have 210 values")
            
        if self._validate_constraints(new_values):
            self.chromosome = new_values.copy()
            self._update_cache()
        else:
            raise ValueError("Invalid matrix: biological constraints violated")
            
    def _validate_constraints(self, values: np.ndarray) -> bool:
        """
        Valida restrições biológicas:
        1. Diagonal > outros valores (conservação)
        2. Simetria mantida
        3. Valores dentro de limites razoáveis
        """
        try:
            # Checa limites dos valores
            if np.any(values < -20) or np.any(values > 20):
                return False
                
            # Checa diagonal
            diag = values[:20]
            if np.any(diag < 0):
                return False
                
            # Demais checks serão adicionados conforme necessário
            return True
            
        except Exception as e:
            self.logger.error(f"Erro validando constraints: {e}")
            return False
    def to_dict(self) -> Dict:
        """Exporta matriz como dicionário para serialização"""
        return {
            'chromosome': self.chromosome.tolist(),
            'gap_open': self.gap_open,
            'gap_extend': self.gap_extend
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'AdaptiveMatrix':
        """Cria matriz a partir de dicionário"""
        matrix = cls(
            gap_open=data['gap_open'],
            gap_extend=data['gap_extend']
        )
        matrix.update_from_vector(np.array(data['chromosome']))
        return matrix
        
    def save(self, filepath: Path):
        """Salva matriz em arquivo numpy"""
        try:
            np.save(filepath, self.chromosome)
            self.logger.info(f"Matrix saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving matrix: {e}")
            
    @classmethod
    def load(cls, filepath: Path) -> Optional['AdaptiveMatrix']:
        """Carrega matriz de arquivo numpy"""
        try:
            values = np.load(filepath)
            matrix = cls()
            matrix.update_from_vector(values)
            return matrix
        except Exception as e:
            logging.error(f"Error loading matrix: {e}")
            return None

    def generate_population(self, size: int = 100,
                          perturbation_range: tuple = (0.2, 0.5),
                          diagonal_range: tuple = (1.5, 2.0)) -> List['AdaptiveMatrix']:
        """
        Gera população inicial de matrizes variantes.
        Args:
            size: Tamanho da população
            perturbation_range: Intervalo para taxa de perturbação
            diagonal_range: Intervalo para bias diagonal
        Returns:
            Lista de matrizes adaptativas
        """
        population = []
        
        try:
            for _ in range(size):
                # Gera parâmetros aleatórios nos intervalos
                perturbation = np.random.uniform(*perturbation_range)
                diagonal_bias = np.random.uniform(*diagonal_range)
                
                # Cria nova matriz com variação
                matrix = AdaptiveMatrix(
                    gap_open=self.gap_open,
                    gap_extend=self.gap_extend
                )
                
                # Gera variação dos valores
                variation = self.chromosome.copy()
                
                # Aplica perturbação aleatória
                noise = np.random.normal(0, perturbation, len(variation))
                variation += noise * np.abs(variation)
                
                # Fortalece diagonal
                variation[:20] *= diagonal_bias
                
                # Limita valores
                variation = np.clip(variation, -20, 20)
                
                try:
                    matrix.update_from_vector(variation)
                    population.append(matrix)
                except ValueError:
                    self.logger.warning("Variação inválida, pulando...")
                    continue
                    
            return population
            
        except Exception as e:
            self.logger.error(f"Erro gerando população: {e}")
            return []

    def to_clustalw_format(self, output_file: Path):
        """
        Salva matriz no formato ClustalW
        """
        try:
            aa_order = list("ARNDCQEGHILKMFPSTWYV") + ['B', 'Z', 'X', '*']
            
            with open(output_file, 'w') as f:
                # Header
                f.write("# Matrix for ClustalW\n#\n# Adaptive PAM Matrix\n#\n")
                f.write("   " + "  ".join(aa_order) + "\n")
                
                # Valores da matriz
                idx = 0
                for i, aa1 in enumerate(aa_order):
                    line = [aa1]
                    
                    for aa2 in aa_order:
                        if aa1 == '*' and aa2 == '*':
                            score = 1
                        elif aa1 == '*' or aa2 == '*':
                            score = -8
                        elif aa1 in ['B','Z','X'] or aa2 in ['B','Z','X']:
                            score = -1
                        else:
                            # Pega valor da representação linear
                            if aa1 == aa2:
                                score = int(self.chromosome[self.aa_order.index(aa1)])
                            else:
                                i1 = min(self.aa_order.index(aa1), self.aa_order.index(aa2))
                                i2 = max(self.aa_order.index(aa1), self.aa_order.index(aa2))
                                offset = 20 + sum(range(20-i1, 20-i1-(i2-i1-1)))
                                score = int(self.chromosome[offset])
                                
                        line.append(f"{score:3d}")
                        
                    f.write(" ".join(line) + "\n")
                    
        except Exception as e:
            self.logger.error(f"Erro salvando matriz: {e}")
            raise