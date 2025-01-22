import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np
from collections import defaultdict
import logging

class LocalSearch:
    """
    Implementa duas buscas locais especializadas para otimização de matriz PAM:
    1. Busca por Conservação: Analisa core blocks do BAliBASE
    2. Busca Estrutural: Usa anotações de estrutura secundária
    
    Args:
        xml_file: Arquivo XML do BAliBASE com anotações
        p_nothing: Probabilidade de não fazer alteração
        p_perturb: Probabilidade de perturbação aleatória  
        p_conservation: Probabilidade de busca por conservação
        p_structural: Probabilidade de busca estrutural
    """
    def __init__(self, xml_file: Path,
                 p_nothing: float,    
                 p_perturb: float,    
                 p_conservation: float,
                 p_structural: float):
                 
        self.logger = logging.getLogger(__name__)
        
        # Probabilidades
        self.p_nothing = p_nothing
        self.p_perturb = p_perturb 
        self.p_conservation = p_conservation
        self.p_structural = p_structural

        # Dados do XML
        self.core_blocks: List[Tuple[int, int]] = []  # Regiões conservadas
        self.pdb_sequences: List[str] = []  # Sequências com estrutura
        self.substitution_freqs: Dict[Tuple[str,str], float] = {}  # Frequências
        self.evolution_hierarchy: Dict[str, float] = {}  # Hierarquia evolutiva
        self.secondary_structure: Dict[int, str] = {}  # Estrutura por posição
        self.motifs: Dict[int, str] = {}  # Motivos funcionais
        self.active_sites: Set[int] = set()  # Sítios ativos
        
        # Ordem dos aminoácidos
        self.aa_order = list("ARNDCQEGHILKMFPSTWYV")
        
        # Propriedades físico-químicas
        self.properties = {
            'hydrophobic': set('AILMFWV'),  # Hidrofóbicos
            'polar': set('STNQ'),          # Polares
            'positive': set('RHK'),        # Carga positiva
            'negative': set('DE'),         # Carga negativa
            'special': set('CGP')          # Especiais (Cys, Gly, Pro)
        }
        
        # Pesos de compatibilidade AA-estrutura
        # Baseados em propensidades estruturais observadas
        self.aa_structure_compat = {
            'H': {  # Hélices alfa - padrão helicoidal
                'A': 1.4, 'L': 1.4, 'M': 1.3, 'E': 1.3, 'K': 1.3,
                'R': 1.2, 'Q': 1.2, 'I': 1.1, 'W': 1.0, 'V': 1.0,
                'F': 0.9, 'Y': 0.9, 'C': 0.8, 'T': 0.8, 'N': 0.8,
                'S': 0.7, 'D': 0.7, 'H': 0.7, 'P': 0.4, 'G': 0.4
            },
            'E': {  # Folhas beta - preservação de dobramento
                'V': 1.4, 'I': 1.4, 'Y': 1.3, 'C': 1.3, 'F': 1.3,
                'T': 1.2, 'W': 1.2, 'L': 1.1, 'A': 0.9, 'M': 0.9,
                'H': 0.8, 'Q': 0.8, 'K': 0.8, 'R': 0.8, 'N': 0.8,
                'S': 0.7, 'P': 0.5, 'D': 0.5, 'E': 0.5, 'G': 0.4
            },
            'L': {  # Loops - maior flexibilidade
                'G': 1.5, 'P': 1.5, 'D': 1.3, 'N': 1.3, 'S': 1.3,
                'E': 1.2, 'K': 1.2, 'Q': 1.2, 'R': 1.1, 'T': 1.1,
                'H': 1.0, 'A': 0.9, 'Y': 0.9, 'C': 0.9, 'M': 0.8,
                'W': 0.7, 'F': 0.7, 'L': 0.7, 'V': 0.6, 'I': 0.6
            }
        }

        # Carrega e valida dados do XML
        if xml_file and xml_file.exists():
            self._load_xml_data(xml_file)
        else:
            self.logger.error(f"Arquivo XML não encontrado: {xml_file}")

    def _load_xml_data(self, xml_file: Path):
        """
        Extrai dados do XML do BAliBASE:
        1. Core blocks e padrões de conservação
        2. Estrutura secundária e motivos
        3. Hierarquia evolutiva e relações
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Identifica sequências PDB 
            pdb_seqs = []
            for seq in root.findall('.//sequence'):
                acc = seq.find(".//accession").text
                if acc.endswith('_A'):
                    pdb_seqs.append(seq)
                    
            if not pdb_seqs:
                raise ValueError("Nenhuma sequência PDB encontrada")
            self.pdb_sequences = [s.find(".//seq-data").text for s in pdb_seqs]
            
            # Extrai core blocks
            colsco = root.find(".//colsco-data")
            if colsco is None or not colsco.text:
                raise ValueError("Core blocks não encontrados")
                
            blocks = []
            start = None
            for i, val in enumerate(colsco.text.split()):
                if val == '1' and start is None:
                    start = i
                elif val != '1' and start is not None:
                    blocks.append((start, i-1))
                    start = None
            if not blocks:
                raise ValueError("Nenhum core block identificado")
            self.core_blocks = blocks
            
            # Extrai estrutura secundária e motivos
            self._extract_structural_info(pdb_seqs[0])
            
            # Calcula hierarquia evolutiva
            self._calculate_conservation_patterns(root)
            
            self.logger.info(
                f"Dados extraídos com sucesso: "
                f"{len(self.core_blocks)} core blocks, "
                f"{len(self.pdb_sequences)} sequências PDB, "
                f"{len(self.secondary_structure)} posições com estrutura"
            )
                           
        except Exception as e:
            self.logger.error(f"Erro carregando XML {xml_file}: {e}")
            raise

    def _extract_structural_info(self, pdb_seq: ET.Element):
        """
        Extrai informações estruturais da primeira sequência PDB:
        - Estrutura secundária (H,E,L)
        - Motivos funcionais 
        - Sítios ativos
        """
        try:
            seq_data = pdb_seq.find('.//seq-data').text
            ftable = pdb_seq.find('.//ftable')
            
            struct_map = {}
            motif_map = {}
            active = set()
            
            if ftable is not None:
                for item in ftable.findall('fitem'):
                    ftype = item.find('ftype').text
                    start = int(item.find('fstart').text) - 1
                    end = int(item.find('fstop').text)
                    
                    # Estrutura secundária
                    if ftype == 'STRUCTURE':
                        struct = item.find('fnote').text[0].upper()
                        if struct in 'HEL':
                            for i in range(start, end):
                                struct_map[i] = struct
                                
                    # Motivos funcionais
                    elif ftype in ('DOMAIN', 'MOTIF', 'BINDING'):
                        for i in range(start, end):
                            motif_map[i] = ftype
                            
                    # Sítios ativos        
                    elif 'ACTIVE' in ftype or 'SITE' in ftype:
                        active.add(start)
                        
            # Infere estrutura para posições sem anotação
            for i, aa in enumerate(seq_data):
                if i not in struct_map and aa != '-':
                    if aa in 'AEKLMR':  # Alta propensidade para hélice
                        struct_map[i] = 'H'
                    elif aa in 'VICYFT':  # Alta propensidade para folha
                        struct_map[i] = 'E'
                    else:  # Default para loop
                        struct_map[i] = 'L'
                        
            self.secondary_structure = struct_map
            self.motifs = motif_map
            self.active_sites = active
            
        except Exception as e:
            self.logger.error(f"Erro extraindo estrutura: {e}")
            raise
    def _calculate_conservation_patterns(self, root):
        """
        Analisa core blocks para estabelecer:
        - Frequências de substituição 
        - Hierarquia evolutiva
        - Padrões de conservação em motivos
        """
        try:
            self.substitution_freqs = defaultdict(float)
            evolution_counts = defaultdict(int)
            motif_freqs = defaultdict(lambda: defaultdict(float))
            
            for block_start, block_end in self.core_blocks:
                sequences = []
                for seq in root.findall('.//sequence'):
                    seq_data = seq.find('.//seq-data').text
                    block = seq_data[block_start:block_end+1]
                    sequences.append(block)

                # Analisa cada coluna
                for i in range(len(sequences[0])):
                    col_aas = [seq[i] for seq in sequences if seq[i] != '-']
                    pos = block_start + i
                    
                    # Frequências de substituição
                    for aa1 in col_aas:
                        for aa2 in col_aas:
                            if aa1 <= aa2:  # Mantém simetria
                                self.substitution_freqs[(aa1, aa2)] += 1
                                
                            # Considera motivos especiais
                            if pos in self.motifs:
                                motif = self.motifs[pos]
                                motif_freqs[motif][(aa1, aa2)] += 1
                                
                        evolution_counts[aa1] += 1

            # Normaliza hierarquia
            total = sum(evolution_counts.values())
            if total > 0:
                self.evolution_hierarchy = {
                    aa: count/total for aa, count in evolution_counts.items()
                }
                
            # Ajusta frequências por motivos
            for motif, freqs in motif_freqs.items():
                motif_total = sum(freqs.values())
                if motif_total > 0:
                    for pair in freqs:
                        freqs[pair] /= motif_total
                        # Aumenta peso de substituições em motivos
                        if pair in self.substitution_freqs:
                            self.substitution_freqs[pair] *= 1.2
                            
        except Exception as e:
            self.logger.error(f"Erro calculando padrões: {e}")
            raise

    def adjust_matrix(self, matrix: np.ndarray, aln_id: str) -> np.ndarray:
        """
        Aplica busca local na matriz usando:
        1. Nada (p_nothing)
        2. Perturbação (p_perturb) 
        3. Busca por conservação (p_conservation)
        4. Busca estrutural (p_structural)
        """
        try:
            if not self._validate_data():
                raise ValueError("Dados XML incompletos")

            new_matrix = matrix.copy()
            scores = []
            
            # Aplica busca local em todas posições
            for i in range(len(matrix)):
                choice = random.random()
                
                if choice < self.p_nothing:
                    continue
                    
                elif choice < (self.p_nothing + self.p_perturb):
                    # Perturbação controlada
                    delta = np.random.normal(0, 0.05)
                    new_matrix[i] += delta
                    
                elif choice < (self.p_nothing + self.p_perturb + self.p_conservation):
                    # Busca por conservação
                    new_matrix = self.conservation_search(new_matrix, i)
                    
                else:
                    # Busca estrutural  
                    new_matrix = self.structural_search(new_matrix, i)
                
                # Valida restrições biológicas
                if not self._validate_constraints(new_matrix):
                    new_matrix[i] = matrix[i]  # Reverte mudança inválida
                else:
                    # Armazena score da modificação
                    aa1, aa2 = self._get_aas_for_position(i)
                    scores.append((self.substitution_freqs.get((aa1, aa2), 0), i))
                    
            # Reforça top 5 melhores modificações
            scores.sort(reverse=True)
            for score, pos in scores[:5]:
                new_matrix[pos] *= 1.1
                
            return new_matrix
            
        except Exception as e:
            self.logger.error(f"Erro na busca local: {e}")
            return matrix

    def conservation_search(self, matrix: np.ndarray, pos: int) -> np.ndarray:
        """
        Busca por Conservação (3.1):
        - Analisa padrões em core blocks
        - Ajusta por frequências observadas
        - Considera motivos e sítios ativos
        """
        try:
            result = matrix.copy()
            
            # Skip se posição não está em core block
            if not any(start <= pos <= end for start, end in self.core_blocks):
                return result
                
            aa1, aa2 = self._get_aas_for_position(pos)
            mult = 1.0
            
            # Fortalece conservação em core blocks
            if aa1 == aa2:  # Diagonal
                mult *= 1.1
                
            # Ajusta por frequências observadas
            key = tuple(sorted([aa1, aa2]))
            freq = self.substitution_freqs.get(key, 0)
            mult *= (1 + 0.1 * freq)
            
            # Considera hierarquia evolutiva
            if aa1 in self.evolution_hierarchy and aa2 in self.evolution_hierarchy:
                h1 = self.evolution_hierarchy[aa1]
                h2 = self.evolution_hierarchy[aa2]
                mult *= (1 + 0.05 * (h1 + h2))
                
            # Reforço para sítios especiais
            if pos in self.active_sites:
                mult *= 1.3  # Maior peso para sítios ativos
            if pos in self.motifs:
                mult *= 1.2  # Peso intermediário para motivos
                
            result[pos] *= mult
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na busca conservação pos {pos}: {e}")
            return matrix

    def structural_search(self, matrix: np.ndarray, pos: int) -> np.ndarray:
        """
        Busca Estrutural (3.2):
        - Usa anotações de estrutura secundária
        - Aplica compatibilidade AA-estrutura
        - Considera propriedades químicas
        """
        try:
            result = matrix.copy()
            
            if pos not in self.secondary_structure:
                return result
                
            struct = self.secondary_structure[pos]
            aa1, aa2 = self._get_aas_for_position(pos)
            mult = 1.0
            
            # Compatibilidade com estrutura
            if aa1 in self.aa_structure_compat[struct]:
                mult *= self.aa_structure_compat[struct][aa1]
            if aa2 in self.aa_structure_compat[struct]:
                mult *= self.aa_structure_compat[struct][aa2]
                
            # Ajusta por tipo estrutural
            if struct == 'H':  # Hélice - mais restritivo
                mult *= 1.2
            elif struct == 'E':  # Folha - intermediário 
                mult *= 1.1
            else:  # Loop - mais flexível
                mult *= 0.9
                
            # Considera propriedades físico-químicas
            for prop, aas in self.properties.items():
                if aa1 in aas and aa2 in aas:  # Mesmo grupo
                    mult *= 1.1
                elif ((aa1 in self.properties['positive'] and aa2 in self.properties['negative']) or
                      (aa1 in self.properties['negative'] and aa2 in self.properties['positive'])):
                    mult *= 0.8  # Penaliza opostos
                    
            result[pos] *= mult
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na busca estrutural pos {pos}: {e}")
            return matrix

    def _get_aas_for_position(self, pos: int) -> Tuple[str, str]:
        """Converte posição linear para par de aminoácidos"""
        try:
            if pos < 0 or pos >= 210:
                raise ValueError(f"Posição {pos} inválida")

            if pos < 20:  # Diagonal
                return (self.aa_order[pos], self.aa_order[pos])

            # Fora da diagonal
            pos -= 20
            i = 0
            while pos >= (19-i):
                pos -= (19-i)
                i += 1
            j = i + 1 + pos
            return (self.aa_order[i], self.aa_order[j])
            
        except Exception as e:
            self.logger.error(f"Erro convertendo posição {pos}: {e}")
            raise

    def _validate_constraints(self, matrix: np.ndarray) -> bool:
        """Valida restrições biológicas"""
        try:
            # Limites globais
            if np.any(matrix < -20) or np.any(matrix > 20):
                return False
                
            # Valida diagonal
            for i in range(20):
                if matrix[i] <= 0:
                    return False
                    
            # Valida simetria implícita
            for i in range(20, len(matrix)):
                aa1, aa2 = self._get_aas_for_position(i)
                if aa1 == aa2 and matrix[i] > matrix[self.aa_order.index(aa1)]:
                    return False
                    
            return True
            
        except Exception:
            return False

    def _validate_data(self) -> bool:
        """Verifica completude dos dados"""
        return all([
            self.core_blocks,
            self.pdb_sequences,
            self.substitution_freqs,
            self.evolution_hierarchy,
            self.secondary_structure
        ])    