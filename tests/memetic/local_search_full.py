import random
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations

class LocalSearch:
    """
    Busca local especializada que implementa dois tipos de busca:
    1. Busca por Conservação em Core Blocks (requisito 3.1)
    2. Busca Estrutural baseada em estrutura secundária (requisito 3.2)
    """
    def __init__(self, xml_file: Path,
                 # Pesos ajustados após análise do BB30002.xml
                 p_nothing: float = 0.2,     # Reduzido para dar mais peso às buscas especializadas
                 p_perturb: float = 0.2,     # Mantido para exploração
                 p_conservation: float = 0.35,# Aumentado pois core blocks são bem definidos
                 p_structural: float = 0.25): # Ajustado baseado na presença de estrutura secundária
        self.xml_file = xml_file
        self.p_nothing = p_nothing
        self.p_perturb = p_perturb
        self.p_conservation = p_conservation
        self.p_structural = p_structural

        # Ordem dos aminoácidos na matriz PAM
        self.aa_order = list("ARNDCQEGHILKMFPSTWYV")
        
        # Dados extraídos do XML
        self.core_blocks: List[Tuple[int,int]] = []
        self.substitution_freqs: Dict[Tuple[str,str], float] = {}
        self.evolution_hierarchy: Dict[str, float] = {}
        self.secondary_structure: Dict[int, str] = {}
        
        # Mapeamento de estrutura secundária por proteína
        self.structure_by_protein: Dict[str, Dict[int, str]] = {}
        
        # Compatibilidade AA-estrutura refinada com dados reais
        self.aa_structure_compat = self._init_aa_structure_compatibility()
        
        # Carrega dados do XML
        self._load_xml_data()

    def _init_aa_structure_compatibility(self) -> Dict[str, Dict[str, float]]:
        """
        Define scores de compatibilidade AA-estrutura baseados em:
        - Propensidades observadas em estruturas cristalográficas
        - Propriedades físico-químicas dos aminoácidos
        - Flexibilidade permitida em cada tipo de estrutura
        
        Atende requisito 3.2: "Compatibilidade dos aminoácidos com cada tipo de estrutura"
        """
        return {
            'H': { # Hélices alfa - padrão helicoidal
                # AAs com alta propensidade para hélices
                'A': 1.4, 'L': 1.4, 'M': 1.3, 'E': 1.3, 'K': 1.3,
                'R': 1.2, 'Q': 1.2, 'I': 1.1, 'W': 1.0, 'V': 1.0,
                # AAs com média propensidade
                'F': 0.9, 'Y': 0.9, 'C': 0.8, 'T': 0.8, 'N': 0.8,
                # AAs que tendem a quebrar hélices
                'S': 0.7, 'D': 0.7, 'H': 0.7, 'P': 0.4, 'G': 0.4
            },
            'E': { # Folhas beta - preservação de dobramento
                # AAs que favorecem folhas
                'V': 1.4, 'I': 1.4, 'Y': 1.3, 'C': 1.3, 'F': 1.3,
                'T': 1.2, 'W': 1.2, 'L': 1.1, 'A': 0.9, 'M': 0.9,
                # AAs neutros para folhas
                'H': 0.8, 'Q': 0.8, 'K': 0.8, 'R': 0.8, 'N': 0.8,
                # AAs que desfavorecem folhas
                'S': 0.7, 'P': 0.5, 'D': 0.5, 'E': 0.5, 'G': 0.4
            },
            'L': { # Loops - maior flexibilidade
                # AAs comuns em loops
                'G': 1.5, 'P': 1.5, 'D': 1.3, 'N': 1.3, 'S': 1.3,
                'E': 1.2, 'K': 1.2, 'Q': 1.2, 'R': 1.1, 'T': 1.1,
                # AAs ocasionalmente em loops
                'H': 1.0, 'A': 0.9, 'Y': 0.9, 'C': 0.9, 'M': 0.8,
                # AAs raros em loops
                'W': 0.7, 'F': 0.7, 'L': 0.7, 'V': 0.6, 'I': 0.6
            }
        }

    def _load_xml_data(self):
        """
        Extrai informações do XML do BAliBASE:
        1. Core blocks e padrões de conservação
        2. Estrutura secundária das proteínas com estrutura conhecida
        3. Hierarquia evolutiva baseada em frequências
        
        Atende requisitos 3.1 e 3.2 sobre extração de informação do XML
        """
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()
            
            # Extrai core blocks da coluna de scores
            # Atende 3.1: "Core blocks são regiões manualmente verificadas"
            colsco = root.find(".//colsco-data").text
            blocks = []
            start = None
            for i, val in enumerate(colsco.split()):
                if val == '1' and start is None:
                    start = i
                elif val != '1' and start is not None:
                    blocks.append((start, i-1))
                    start = None
            self.core_blocks = blocks
            
            # Extrai estruturas secundárias
            # Atende 3.2: "utiliza as anotações de estrutura secundária"
            for seq in root.findall('.//sequence'):
                acc = seq.find(".//accession").text
                if acc.endswith('_A'): # Estrutura PDB
                    seq_data = seq.find('.//seq-data').text
                    struct_map = {}
                    
                    # Usa ftable quando disponível
                    ftable = seq.find('.//ftable')
                    if ftable is not None:
                        for fitem in ftable.findall('fitem'):
                            ftype = fitem.find('ftype').text
                            if ftype == 'STRUCTURE':
                                start = int(fitem.find('fstart').text) - 1
                                end = int(fitem.find('fstop').text)
                                struct = fitem.find('fnote').text[0] # H, E ou L
                                for i in range(start, end):
                                    struct_map[i] = struct
                    
                    # Infere estrutura de padrões de sequência quando necessário
                    pos = 0
                    for i, aa in enumerate(seq_data):
                        if aa != '-':
                            if i not in struct_map:
                                # Heurística baseada em propensidades
                                if aa in 'AEKLMR':
                                    struct_map[i] = 'H'
                                elif aa in 'VICYFT':
                                    struct_map[i] = 'E'
                                else:
                                    struct_map[i] = 'L'
                            pos += 1
                            
                    self.structure_by_protein[acc] = struct_map
                    
                    # Usa primeira estrutura como referência
                    if not self.secondary_structure:
                        self.secondary_structure = struct_map
            
            # Calcula frequências de substituição e hierarquia evolutiva
            # Atende 3.1: "Calcula frequências de substituição nestas regiões"
            self.substitution_freqs = defaultdict(float)
            evolution_counts = defaultdict(int)
            
            for block_start, block_end in self.core_blocks:
                sequences = []
                for seq in root.findall('.//sequence'):
                    seq_data = seq.find('.//seq-data').text
                    block_aas = seq_data[block_start:block_end+1]
                    sequences.append(block_aas)
                
                # Conta substituições em core blocks
                for i in range(len(sequences[0])):
                    col_aas = [seq[i] for seq in sequences if seq[i] != '-']
                    for aa1, aa2 in combinations(col_aas, 2):
                        key = tuple(sorted([aa1, aa2]))
                        self.substitution_freqs[key] += 1
                        evolution_counts[aa1] += 1
                        evolution_counts[aa2] += 1
            
            # Normaliza frequências evolutivas
            # Atende 3.1: "Mantém a hierarquia evolutiva observada"
            total = sum(evolution_counts.values())
            if total > 0:
                self.evolution_hierarchy = {
                    aa: count/total for aa, count in evolution_counts.items()
                }
                
        except Exception as e:
            print(f"Erro carregando XML {self.xml_file}: {e}")

    def _get_aas_for_position(self, pos: int) -> Tuple[str, str]:
        """
        Converte posição na matriz linear para par de aminoácidos.
        Implementação correta baseada na ordem dos AAs na PAM.
        
        Args:
            pos: posição na matriz linear (0-209)
        Returns:
            Tupla com os dois aminoácidos correspondentes
        """
        if pos < 0 or pos >= 210:
            raise ValueError(f"Posição {pos} inválida")
            
        if pos < 20:  # Diagonal
            return (self.aa_order[pos], self.aa_order[pos])
            
        # Posições fora da diagonal
        pos -= 20  # Desconta diagonal
        i = 0
        while pos >= (19-i):
            pos -= (19-i)
            i += 1
        j = i + 1 + pos
        return (self.aa_order[i], self.aa_order[j])

    def adjust_matrix(self, matrix: np.ndarray, aln_id: str) -> np.ndarray:
        """
        Aplica busca local especializada na matriz.
        
        Atende os dois tipos principais de busca (3.1 e 3.2):
        - Busca por conservação em core blocks
        - Busca estrutural em regiões com estrutura secundária
        """
        if not self._validate_data():
            print("Dados XML incompletos")
            return matrix
            
        new_matrix = matrix.copy()
        
        for i in range(len(matrix)):
            choice = random.random()
            
            if choice < self.p_nothing:
                continue
                
            elif choice < (self.p_nothing + self.p_perturb):
                # Perturbação aleatória controlada
                delta = np.random.normal(0, 0.1)
                new_matrix[i] += delta
                
            elif choice < (self.p_nothing + self.p_perturb + self.p_conservation):
                # Busca por conservação
                self._conservation_search(new_matrix, i)
                
            else:
                # Busca estrutural
                self._structural_search(new_matrix, i)
            
            # Validações biológicas
            self._apply_biological_constraints(new_matrix, i)
                
        return new_matrix

    def _conservation_search(self, matrix: np.ndarray, pos: int):
        """
        Implementa busca por conservação em core blocks.
        
        Atende requisito 3.1:
        - Usa core blocks do BAliBASE
        - Considera frequências de substituição
        - Mantém hierarquia evolutiva
        """
        try:
            # Verifica se posição está em core block
            in_core = any(start <= pos <= end for start, end in self.core_blocks)
            
            if in_core:
                aa1, aa2 = self._get_aas_for_position(pos)
                
                # Fortalece conservação em core blocks
                matrix[pos] *= 1.2
                
                # Ajusta por frequências observadas
                key = tuple(sorted([aa1, aa2]))
                if key in self.substitution_freqs:
                    freq = self.substitution_freqs[key]
                    matrix[pos] *= (1 + 0.15 * freq)  # Aumentado peso da frequência
                    
                # Considera hierarquia evolutiva
                if aa1 in self.evolution_hierarchy and aa2 in self.evolution_hierarchy:
                    h1 = self.evolution_hierarchy[aa1]
                    h2 = self.evolution_hierarchy[aa2]
                    matrix[pos] *= (1 + 0.1 * (h1 + h2))
                    
        except Exception as e:
            print(f"Erro na busca por conservação pos {pos}: {e}")

    def _structural_search(self, matrix: np.ndarray, pos: int):
        """
        Implementa busca baseada em estrutura secundária.
        
        Atende requisito 3.2:
        - Considera tipo de estrutura (H, E, L)
        - Usa compatibilidade AA-estrutura
        - Aplica flexibilidade apropriada
        """
        try:
            aa1, aa2 = self._get_aas_for_position(pos)
            struct = self.secondary_structure.get(pos)
            
            if struct in ('H', 'E', 'L'):
                # Pega compatibilidades
                compat1 = self.aa_structure_compat[struct].get(aa1, 0.5)
                compat2 = self.aa_structure_compat[struct].get(aa2, 0.5)
                
                # Ajusta baseado no tipo de estrutura
                if struct == 'H':  # Hélices - mais conservador
                    matrix[pos] *= (compat1 * compat2 * 1.2)
                elif struct == 'E':  # Folhas - intermediário
                    matrix[pos] *= (compat1 * compat2 * 1.1)
                else:  # Loops - mais flexível
                    matrix[pos] *= (compat1 * compat2 * 0.9)
                    
        except Exception as e:
            print(f"Erro na busca estrutural pos {pos}: {e}")

    def _apply_biological_constraints(self, matrix: np.ndarray, pos: int):
        """
        Aplica restrições biológicas à matriz:
        - Mantém simetria
        - Limita valores dentro de faixas aceitáveis
        - Garante que diagonal seja mais conservada
        - Preserva relações químicas entre aminoácidos
        """
        try:
            # Limites básicos
            matrix[pos] = np.clip(matrix[pos], -20, 20)
            
            # Diagonal deve ser mais conservada
            aa1, aa2 = self._get_aas_for_position(pos)
            if aa1 == aa2:  # Posição diagonal
                matrix[pos] = max(matrix[pos], 1.0)  # Sempre positivo
                
            # Preserva relações químicas
            self._adjust_by_chemical_properties(matrix, pos, aa1, aa2)
            
        except Exception as e:
            print(f"Erro aplicando restrições pos {pos}: {e}")

    def _adjust_by_chemical_properties(self, matrix: np.ndarray, pos: int, aa1: str, aa2: str):
        """
        Ajusta scores baseado em propriedades químicas dos aminoácidos
        """
        # Grupos químicos
        hydrophobic = set('AILMFWV')
        polar = set('STNQ')
        positive = set('RHK')
        negative = set('DE')
        special = set('CGP')
        
        # Mesmo grupo -> favorece substituição
        if (aa1 in hydrophobic and aa2 in hydrophobic) or \
           (aa1 in polar and aa2 in polar) or \
           (aa1 in positive and aa2 in positive) or \
           (aa1 in negative and aa2 in negative):
            matrix[pos] *= 1.2
            
        # Grupos opostos -> penaliza
        elif (aa1 in positive and aa2 in negative) or \
             (aa1 in negative and aa2 in positive) or \
             (aa1 in hydrophobic and (aa2 in polar or aa2 in charged)):
            matrix[pos] *= 0.8
            
    def _validate_data(self) -> bool:
        """Valida se temos dados suficientes para busca local"""
        return all([
            self.core_blocks,
            self.substitution_freqs,
            self.evolution_hierarchy,
            self.secondary_structure,
            self.aa_structure_compat
        ])