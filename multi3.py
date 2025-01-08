import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import logging
from math import comb
import math
import numpy as np
import csv 

# ========================================================================#
#                                 MULTIONE                                #
# ========================================================================#
#
# Este script tem a finalidade de:
#  1) Para cada arquivo .tfa (entrada) do BAliBASE:
#       - Gerar alinhamentos via ClustalW (em formato .aln)
#       - Gerar alinhamentos via MUSCLE (em formato .aln)
#       - Converter o arquivo BAliBASE Gold (que está em .msf) 
#         para .aln (Clustal), se ainda não houver, apenas 
#         para poder ler via Biopython.
#
#  2) "Auto-avaliar" cada um desses três alinhamentos
#     (ClustalW, MUSCLE, BAliBASE) de forma completamente interna, 
#     isto é, sem qualquer comparação cruzada.
#     As métricas adotadas são:
#
#       a) SP = Sum of Pairs (interno) - MINIMIZAÇÃO
#          => Calcula, para cada coluna, quantos pares de
#             sequências têm resíduos diferentes ou gaps.
#          => Score 0 para pares idênticos, 1 para diferentes/gaps.
#          => Mostra valores brutos e normalizados [0..1].
#
#       b) WSP = Weighted Sum of Pairs (interno) - MAXIMIZAÇÃO
#          => Implementado segundo **Gotoh (1994)**, construindo
#             uma árvore (Neighbor Joining) e aplicando a recursão
#             (wsp2_gotoh) para somar as colunas pontuadas.
#          => Mantém comportamento original de maximização.
#          => Mostra valores brutos e normalizados [0..1].
#
#  3) Ao final, gera dois arquivos CSV separados para SP e WSP:
#       - sp_scores.csv  (minimização)
#       - wsp_scores.csv (maximização)
#     Cada arquivo contém 3 linhas por instância:
#       - ClustalW
#       - MUSCLE
#       - BAliBASE (referência)
#
# =========================================================================

LOG_DIR = Path("/home/augusto/projects/multiOne/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "alignment_pipeline.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Ajuste conforme seu ambiente
CLUSTALW_PATH = "/home/augusto/projects/multiOne/clustalw-2.1/src/clustalw2"
MUSCLE_PATH   = "/home/augusto/projects/multiOne/muscle-5.3/src/muscle-linux"

# Estrutura para organizar scores brutos e normalizados
class AlignmentScores(NamedTuple):
    sp_raw: float     # SP score bruto (quantidade de diferenças/gaps)
    sp_norm: float    # SP normalizado [0,1] (0 = perfeito, 1 = pior caso)
    wsp_raw: float    # WSP score bruto (quantidade de matches ponderados)
    wsp_norm: float   # WSP normalizado [0,1] (1 = perfeito, 0 = pior caso)

# =========================================================================
#                   FUNÇÕES BASE PARA SP (MINIMIZAÇÃO)
# =========================================================================

def compute_max_score_for_sp(alignment: MultipleSeqAlignment) -> float:
    """
    Calcula máximo teórico de diferenças para SP.
    Específico para SP como problema de minimização.
    
    - max_pairs = N*(N-1)/2 onde N = num_seqs
    - total = max_pairs * comprimento
    - Representa pior caso possível (tudo diferente/gap)
    """
    nseqs = len(alignment)
    length = alignment.get_alignment_length()
    
    # Máximo de pares diferentes possíveis por coluna
    max_pairs = nseqs * (nseqs - 1) / 2
    
    # Total considerando todas as colunas
    return max_pairs * length

def pairwise_distance_for_sp(seq1: str, seq2: str) -> float:
    """
    Calcula distância entre duas sequências para SP.
    
    Retorna:
    - 0.0 = sequências idênticas (melhor caso)
    - 1.0 = sequências totalmente diferentes (pior caso)
    """
    if len(seq1) != len(seq2):
        return 1.0
        
    differences = total = 0
    for c1, c2 in zip(seq1, seq2):
        if c1 != '-' or c2 != '-':  # Ignora coluna só com gaps
            total += 1
            # Conta 1 para diferente/gap
            if c1 != c2 or c1 == '-' or c2 == '-':
                differences += 1
                
    return differences/total if total > 0 else 1.0

# =========================================================================
#                   FUNÇÕES BASE PARA WSP (MAXIMIZAÇÃO)
# =========================================================================

def compute_max_score_for_wsp(alignment: MultipleSeqAlignment) -> float:
    """
    Calcula máximo teórico de matches ponderados para WSP.
    Específico para WSP como problema de maximização.
    Mantém implementação original de Gotoh.
    
    - max_pairs = N*(N-1)/2 onde N = num_seqs
    - total = max_pairs * comprimento
    - Representa melhor caso possível (tudo igual)
    """
    nseqs = len(alignment)
    length = alignment.get_alignment_length()
    
    # Máximo de matches possíveis por coluna
    max_col_score = nseqs * (nseqs - 1) / 2
    
    # Total considerando todas as colunas
    return max_col_score * length

def pairwise_distance_for_wsp(seq1: str, seq2: str) -> float:
    """
    Calcula distância evolutiva para WSP.
    Usado para construir árvore NJ.
    Mantém implementação original de Gotoh.
    
    Retorna:
    1.0 - (matches/total) ignorando gaps
    - 1.0 = sequências totalmente diferentes
    - 0.0 = sequências idênticas
    """
    if len(seq1) != len(seq2):
        return 1.0
        
    matches = total = 0
    for c1, c2 in zip(seq1, seq2):
        if c1 != '-' and c2 != '-':
            total += 1
            if c1 == c2:
                matches += 1
                
    return 1.0 - (matches/total if total > 0 else 0.0)
# =========================================================================
#                   IMPLEMENTAÇÃO DO SP (MINIMIZAÇÃO)
# =========================================================================

def calculate_raw_sp_score(alignment: MultipleSeqAlignment) -> float:
    """
    Calcula SP score sem normalização (MINIMIZAÇÃO):
    
    Para cada coluna do alinhamento:
      * Compara todos os pares possíveis de sequências
      * Soma 0 para matches (resíduos idênticos)
      * Soma 1 para mismatches (resíduos diferentes)
      * Soma 1 para gaps (qualquer par com gap)
      
    O score final é a soma de todas as diferenças/gaps encontrados.
    Quanto menor o score, melhor o alinhamento:
      - 0 = alinhamento perfeito (todas sequências idênticas)
      - max_score = pior caso (todas sequências diferentes ou gaps)
    """
    nseqs = len(alignment)
    length = alignment.get_alignment_length()
    if nseqs < 2 or length == 0:
        return 0.0

    # Conta diferenças totais no alinhamento
    difference_count = 0
    for col in range(length):
        for i in range(nseqs):
            for j in range(i+1, nseqs):
                c1 = alignment[i, col]
                c2 = alignment[j, col]
                # Score 1 para diferentes ou gaps (minimização)
                if c1 != c2 or c1 == '-' or c2 == '-':
                    difference_count += 1

    return float(difference_count)

def calculate_normalized_sp_score(alignment: MultipleSeqAlignment) -> float:
    """
    Normaliza SP score para [0,1] (MINIMIZAÇÃO):
    
    Score = num_diferenças / max_diferenças_possível
    
    Propriedades do score normalizado:
    - 0.0 = alinhamento perfeito (nenhuma diferença)
    - 1.0 = pior caso (máximo de diferenças)
    
    A normalização permite:
    - Comparar alinhamentos de tamanhos diferentes
    - Manter escala consistente [0,1]
    - Preservar direção de minimização (menor=melhor)
    """
    raw_score = calculate_raw_sp_score(alignment)
    max_score = compute_max_score_for_sp(alignment)
    return raw_score / max_score if max_score > 0 else 1.0

def evaluate_sp_only(alignment: MultipleSeqAlignment) -> Tuple[float, float]:
    """
    Avalia alinhamento usando apenas métrica SP (MINIMIZAÇÃO).
    
    Retorna:
    - sp_raw: Número total de diferenças encontradas
    - sp_norm: Score normalizado [0,1] (0 = melhor)
    
    Esta função isola completamente a avaliação SP da WSP,
    permitindo calcular apenas SP quando necessário.
    """
    try:
        sp_raw = calculate_raw_sp_score(alignment)
        sp_norm = calculate_normalized_sp_score(alignment)
        return sp_raw, sp_norm
        
    except Exception as e:
        logger.error(f"Erro no cálculo do SP: {e}")
        return 0.0, 1.0  # Retorna pior caso possível em caso de erro

def compare_sp_scores(score1: float, score2: float) -> int:
    """
    Compara dois SP scores normalizados (MINIMIZAÇÃO).
    
    Retorna:
    -  1: score1 é pior que score2
    -  0: scores são iguais
    - -1: score1 é melhor que score2
    
    Lembre-se: para SP, menor = melhor.
    """
    if abs(score1 - score2) < 1e-10:  # Tolerância para floating point
        return 0
    return 1 if score1 > score2 else -1

def format_sp_score(raw_score: float, norm_score: float) -> str:
    """
    Formata scores SP para exibição com direção de otimização.
    
    Exemplo:
    "SP - Bruto: 123.4500, Normalizado [0,1]: 0.8234 (menor=melhor)"
    """
    return (
        f"SP  - Bruto: {raw_score:.4f}, "
        f"Normalizado [0,1]: {norm_score:.4f} "
        "(menor=melhor)"
    )
# =========================================================================
#                   IMPLEMENTAÇÃO DO WSP (MAXIMIZAÇÃO)
# =========================================================================

class TreeNode:
    """
    Nó da árvore filogenética para cálculo do WSP.
    Implementação mantém abordagem original de Gotoh (1994).
    
    Cada nó mantém:
    - Identificador único
    - Status (folha/interno)
    - Lista de filhos com pesos dos ramos
    - Sequência (apenas folhas)
    - Perfis P e Q conforme artigo original
    """
    def __init__(self, node_id: str, is_leaf: bool = False):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.children: List[Tuple['TreeNode', float]] = []  # [(filho, peso),...]
        self.sequence: Optional[str] = None
        self.profile_p: Dict[int, Dict[str, float]] = {}  # posição -> {resíduo -> freq}
        self.profile_q: Dict[int, Dict[str, float]] = {}  # posição -> {resíduo -> freq}

def sequence_to_profile(seq: str) -> Dict[int, Dict[str, float]]:
    """
    Converte sequência em perfil de frequências.
    Usado para inicialização de folhas na árvore WSP.
    
    Para cada posição da sequência:
    - Cria distribuição sobre resíduos válidos
    - Atribui frequência 1.0 ao resíduo observado
    - Atribui frequência 0.0 aos demais resíduos
    
    Este perfil é usado nos cálculos recursivos do WSP
    para computar similaridade ponderada entre sequências.
    """
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY-')
    profile = {}
    for i, c in enumerate(seq):
        freqs = {ch: 0.0 for ch in valid_chars}
        if c in valid_chars:
            freqs[c] = 1.0
        profile[i] = freqs
    return profile

def merge_profiles(p1: Dict[int, Dict[str, float]], 
                  p2: Dict[int, Dict[str, float]], 
                  weight: float) -> Dict[int, Dict[str, float]]:
    """
    Combina dois perfis com peso aplicado ao segundo.
    Mantém lógica original de Gotoh para WSP.
    
    O peso vem do comprimento do ramo na árvore NJ e
    influencia quanto cada perfil contribui para o resultado.
    Este método é fundamental para a recursão bottom-up
    que calcula o WSP ponderado.
    """
    result = {}
    all_pos = set(p1.keys()) | set(p2.keys())
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY-')
    
    for pos in all_pos:
        freqs = {ch: 0.0 for ch in valid_chars}
        for ch in valid_chars:
            v1 = p1.get(pos, {}).get(ch, 0.0)
            v2 = p2.get(pos, {}).get(ch, 0.0)
            freqs[ch] = weight * (v1 + v2)
        result[pos] = freqs
    return result

def dot_product(p1: Dict[int, Dict[str, float]], 
                p2: Dict[int, Dict[str, float]]) -> float:
    """
    Calcula produto escalar entre dois perfis.
    Parte essencial do cálculo do WSP conforme Gotoh.
    
    Este produto mede similaridade entre distribuições
    de resíduos em cada posição, considerando as 
    frequências ponderadas pelos pesos da árvore.
    """
    score = 0.0
    for pos in set(p1.keys()) & set(p2.keys()):
        for ch in set(p1[pos].keys()) & set(p2[pos].keys()):
            score += p1[pos][ch] * p2[pos][ch]
    return score

def neighbor_joining_gotoh(dist_matrix: np.ndarray, names: List[str]) -> Optional[TreeNode]:
    """
    Implementa Neighbor Joining com melhorias para WSP.
    Mantém abordagem original de Gotoh (1994).
    
    Extensões específicas:
    - Controle rigoroso de índices da matriz
    - Preservação de valores na expansão
    - Tratamento robusto de erros
    """
    try:
        n_initial = len(names)
        if n_initial < 2:
            return None
            
        D = dist_matrix.copy()
        old_size = D.shape[0]
        nodes = [TreeNode(name, is_leaf=True) for name in names]
        active = list(range(n_initial))
        
        while len(active) > 1:
            n_active = len(active)
            
            # Soma de linhas para matriz Q
            r = np.array([sum(D[i,j] for j in active if j != i) 
                         for i in active])
            
            # Encontra par mais próximo
            min_q = float('inf')
            min_i = min_j = -1
            
            for ai, i in enumerate(active[:-1]):
                for j in active[ai + 1:]:
                    q = (n_active - 2) * D[i,j] - r[active.index(i)] - r[active.index(j)]
                    if q < min_q:
                        min_q = q
                        min_i, min_j = i, j
            
            if min_i == -1:
                return None
                
            # Calcula comprimentos dos ramos
            d_ij = D[min_i, min_j]
            if n_active > 2:
                d_i = 0.5 * d_ij + (r[active.index(min_i)] - r[active.index(min_j)]) / (2 * (n_active - 2))
            else:
                d_i = 0.5 * d_ij
            d_j = d_ij - d_i
            
            # Cria novo nó interno
            new_node = TreeNode(f"internal_{len(nodes)}")
            new_node.children.append((nodes[min_i], d_i))
            new_node.children.append((nodes[min_j], d_j))
            nodes.append(new_node)
            
            # Expande matriz de distâncias
            new_idx = len(nodes) - 1
            if new_idx >= old_size:
                D_new = np.zeros((old_size + 1, old_size + 1))
                D_new[:old_size, :old_size] = D
                D = D_new
                old_size += 1
            
            # Calcula distâncias ao novo nó
            for k in active:
                if k not in (min_i, min_j):
                    dist_k = 0.5 * (D[min_i,k] + D[min_j,k] - D[min_i,min_j])
                    D[new_idx,k] = D[k,new_idx] = dist_k
            
            active.remove(min_i)
            active.remove(min_j)
            active.append(new_idx)
        
        return nodes[-1]
        
    except Exception as e:
        logger.error(f"Erro durante Neighbor Joining: {e}")
        return None

def wsp2_gotoh(node: TreeNode) -> Optional[float]:
    """
    Implementa recursão bottom-up do WSP (Gotoh 1994).
    Mantém comportamento original de maximização.
    
    Para cada nó:
    1. Se folha:
       - Converte sequência em perfis P=Q
       - Retorna 0.0 (base da recursão)
       
    2. Se interno:
       - Processa filhos recursivamente
       - Soma scores parciais
       - Combina perfis com pesos
       - Adiciona pontuação da combinação
    """
    try:
        if node is None:
            return None
            
        if node.is_leaf:
            if node.sequence is None:
                return None
            node.profile_p = sequence_to_profile(node.sequence)
            node.profile_q = sequence_to_profile(node.sequence)
            return 0.0
            
        total_score = 0.0
        
        for child, weight in node.children:
            child_score = wsp2_gotoh(child)
            if child_score is None:
                return None
            total_score += child_score
            
            if not node.profile_p:
                node.profile_p = merge_profiles({}, child.profile_p, weight)
                node.profile_q = merge_profiles({}, child.profile_q, weight)
            else:
                partial = weight * dot_product(node.profile_p, child.profile_q)
                total_score += partial
                
                node.profile_p = merge_profiles(node.profile_p, child.profile_p, weight)
                node.profile_q = merge_profiles(node.profile_q, child.profile_q, weight)
                
        return total_score
        
    except Exception as e:
        logger.error(f"Erro na recursão WSP: {e}")
        return None

def compute_wsp_gotoh(alignment: MultipleSeqAlignment, 
                     normalize: bool = True, 
                     F: float = 1.15) -> float:
    """
    Implementa WSP conforme Gotoh (1994).
    Mantém comportamento original de maximização.
    
    Passos principais:
    1. Constrói árvore filogenética via NJ
    2. Associa sequências às folhas
    3. Calcula WSP via recursão bottom-up
    4. Normaliza usando alinhamento perfeito
    5. Aplica fator de equalização F
    
    O score final indica qualidade do alinhamento:
    - Maior valor = melhor alinhamento
    - Score normalizado entre [0,1]
    """
    try:
        if alignment is None or len(alignment) < 2:
            return 0.0
            
        # Constrói matriz de distância
        n_seqs = len(alignment)
        dist_matrix = np.zeros((n_seqs, n_seqs))
        
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                dist = pairwise_distance_for_wsp(alignment[i], alignment[j])
                dist_matrix[i,j] = dist_matrix[j,i] = dist
        
        # Constrói árvore via NJ
        tree = neighbor_joining_gotoh(dist_matrix, [f"seq_{i}" for i in range(n_seqs)])
        if tree is None:
            return 0.0
            
        # Associa sequências às folhas
        stack = [(tree, None)]
        while stack:
            node, parent = stack.pop()
            if node.is_leaf:
                idx = int(node.node_id.split('_')[1])
                node.sequence = str(alignment[idx].seq)
            for child, _ in node.children:
                stack.append((child, node))
                
        # Calcula WSP do alinhamento real
        wsp_raw = wsp2_gotoh(tree)
        if wsp_raw is None:
            return 0.0
            
        if not normalize:
            return wsp_raw * F
            
        # Calcula WSP do alinhamento perfeito
        perfect_seq = "A" * alignment.get_alignment_length()
        
        perfect_tree = neighbor_joining_gotoh(dist_matrix, [f"seq_{i}" for i in range(n_seqs)])
        if perfect_tree is None:
            return wsp_raw * F
            
        stack = [(perfect_tree, None)]
        while stack:
            node, parent = stack.pop()
            if node.is_leaf:
                node.sequence = perfect_seq
            for child, _ in node.children:
                stack.append((child, node))
                
        wsp_perfect = wsp2_gotoh(perfect_tree)
        if wsp_perfect is None or wsp_perfect == 0.0:
            return wsp_raw * F
            
        # Normaliza e aplica fator de equalização
        wsp_normalized = (wsp_raw / wsp_perfect) * F
        
        return wsp_normalized
        
    except Exception as e:
        logger.error(f"Erro no cálculo do WSP: {e}")
        return 0.0

def evaluate_wsp_only(alignment: MultipleSeqAlignment) -> Tuple[float, float]:
    """
    Avalia alinhamento usando apenas métrica WSP (MAXIMIZAÇÃO).
    
    Retorna:
    - wsp_raw: Score bruto do alinhamento
    - wsp_norm: Score normalizado [0,1] (1 = melhor)
    
    Esta função isola completamente a avaliação WSP da SP,
    permitindo calcular apenas WSP quando necessário.
    """
    try:
        wsp_raw = compute_wsp_gotoh(alignment, normalize=False)
        wsp_norm = compute_wsp_gotoh(alignment, normalize=True)
        return wsp_raw, wsp_norm
        
    except Exception as e:
        logger.error(f"Erro no cálculo do WSP: {e}")
        return 0.0, 0.0  # Retorna pior caso possível em caso de erro

def format_wsp_score(raw_score: float, norm_score: float) -> str:
    """
    Formata scores WSP para exibição com direção de otimização.
    
    Exemplo:
    "WSP - Bruto: 123.4500, Normalizado [0,1]: 0.8234 (maior=melhor)"
    """
    return (
        f"WSP - Bruto: {raw_score:.4f}, "
        f"Normalizado [0,1]: {norm_score:.4f} "
        "(maior=melhor)"
    )
class SequenceAlignmentPipeline:
    """
    Pipeline principal para avaliação de alinhamentos múltiplos.
    Implementa avaliação separada de SP (minimização) e WSP (maximização).
    """
    def __init__(self, balibase_dir: Path, reference_dir: Path, results_dir: Path):
        self.balibase_dir = balibase_dir
        self.reference_dir = reference_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.clustalw_results_dir = self.results_dir / "clustalw"
        self.muscle_results_dir = self.results_dir / "muscle"
        self.clustalw_results_dir.mkdir(parents=True, exist_ok=True)
        self.muscle_results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline inicializado - Avaliação de alinhamentos múltiplos")
        logger.info("SP (minimização) e WSP (maximização) implementados separadamente")
        logger.info(f"Dir entradas: {self.balibase_dir}")
        logger.info(f"Dir referências: {self.reference_dir}")
        logger.info(f"Dir resultados: {self.results_dir}")

    def run_clustalw(self, input_file: Path) -> Path:
        """Executa alinhamento via ClustalW."""
        output_file = self.clustalw_results_dir / f"{input_file.stem}_clustalw.aln"
        cmd = [
            CLUSTALW_PATH,
            "-INFILE=" + str(input_file),
            "-ALIGN",
            "-OUTPUT=CLUSTAL",
            "-OUTFILE=" + str(output_file)
        ]
        logger.info(f"[CLUSTALW] Gerando alinhamento: {input_file.name} -> {output_file.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise Exception("[CLUSTALW] Falhou, arquivo de saída vazio ou ausente")
        logger.info(f"[CLUSTALW] Alinhamento concluído: {output_file}")
        return output_file

    def run_muscle(self, input_file: Path) -> Path:
        """Executa alinhamento via MUSCLE."""
        output_fasta = self.muscle_results_dir / f"{input_file.stem}_temp.fa"
        output_file = self.muscle_results_dir / f"{input_file.stem}_muscle.aln"

        logger.info(f"[MUSCLE] Gerando FASTA intermediário: {input_file.name}")
        muscle_cmd = [
            str(MUSCLE_PATH),
            "-align", str(input_file),
            "-output", str(output_fasta)
        ]
        subprocess.run(muscle_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not output_fasta.exists() or output_fasta.stat().st_size == 0:
            raise Exception("[MUSCLE] Falhou na geração do FASTA")

        logger.info(f"[MUSCLE] Convertendo para CLUSTAL: {output_file.name}")
        alignments = AlignIO.read(output_fasta, "fasta")
        AlignIO.write(alignments, output_file, "clustal")

        logger.info(f"[MUSCLE] Removendo FASTA temporário")
        output_fasta.unlink()

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise Exception("[MUSCLE] Falhou na conversão para CLUSTAL")

        logger.info(f"[MUSCLE] Alinhamento concluído: {output_file}")
        return output_file

    def convert_msf_to_aln(self, msf_file: Path) -> Path:
        """Converte arquivo MSF do BAliBASE para formato ALN."""
        aln_file = msf_file.with_suffix(".aln")
        if aln_file.exists() and aln_file.stat().st_size > 0:
            logger.info(f"[BAliBASE] Referência já convertida: {msf_file.name}")
            return aln_file

        cmd = [
            "seqret",
            "-sequence", str(msf_file),
            "-outseq", str(aln_file),
            "-osformat2", "clustal"
        ]
        logger.info(f"[BAliBASE] Convertendo MSF para CLUSTAL: {msf_file.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not aln_file.exists() or aln_file.stat().st_size == 0:
            raise Exception(f"[BAliBASE] Falha na conversão MSF->CLUSTAL: {msf_file.name}")

        logger.info(f"[BAliBASE] Conversão concluída: {aln_file}")
        return aln_file

    def evaluate_alignment(self, aln_file: Path) -> Optional[Dict[str, float]]:
        """
        Avalia um alinhamento usando SP e WSP separadamente.
        Mantém direções de otimização distintas para cada métrica.
        """
        try:
            logger.info(f"\nAvaliando alinhamento: {aln_file}")
            alignment = AlignIO.read(aln_file, "clustal")
            
            # Avalia SP (minimização)
            sp_raw, sp_norm = evaluate_sp_only(alignment)
            
            # Avalia WSP (maximização)
            wsp_raw, wsp_norm = evaluate_wsp_only(alignment)
            
            # Log detalhado com direções de otimização explícitas
            logger.info("Scores calculados:")
            logger.info("  " + format_sp_score(sp_raw, sp_norm))
            logger.info("  " + format_wsp_score(wsp_raw, wsp_norm))
            
            return {
                "sp_raw": sp_raw,
                "sp_norm": sp_norm,
                "wsp_raw": wsp_raw,
                "wsp_norm": wsp_norm
            }
            
        except Exception as e:
            logger.error(f"Erro ao avaliar {aln_file}: {e}")
            return None

    def _save_sp_scores(self, results: List[Dict], output_file: Path) -> None:
        """
        Salva scores SP em arquivo separado.
        Mantém apenas métricas de minimização.
        """
        logger.info(f"Salvando scores SP (minimização): {output_file}")
        
        fieldnames = ["sequence", "method", "sp_raw", "sp_norm"]
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in results:
                seq_id = item["sequence"]
                for i, method in enumerate(item["method"]):
                    writer.writerow({
                        "sequence": seq_id,
                        "method": method,
                        "sp_raw": item["sp_raw"][i],
                        "sp_norm": item["sp_norm"][i]
                    })

    def _save_wsp_scores(self, results: List[Dict], output_file: Path) -> None:
        """
        Salva scores WSP em arquivo separado.
        Mantém apenas métricas de maximização.
        """
        logger.info(f"Salvando scores WSP (maximização): {output_file}")
        
        fieldnames = ["sequence", "method", "wsp_raw", "wsp_norm"]
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in results:
                seq_id = item["sequence"]
                for i, method in enumerate(item["method"]):
                    writer.writerow({
                        "sequence": seq_id,
                        "method": method,
                        "wsp_raw": item["wsp_raw"][i],
                        "wsp_norm": item["wsp_norm"][i]
                    })

    def save_results(self, results: List[Dict]) -> None:
        """
        Salva resultados em arquivos separados para SP e WSP.
        """
        if not results:
            logger.error("Nenhum resultado para salvar")
            return
            
        # Salva SP scores (minimização)
        sp_file = self.results_dir / "sp_scores.csv"
        self._save_sp_scores(results, sp_file)
        
        # Salva WSP scores (maximização)
        wsp_file = self.results_dir / "wsp_scores.csv"
        self._save_wsp_scores(results, wsp_file)
        
        logger.info("Resultados salvos com sucesso")

    def run_pipeline(self) -> None:
        """
        Executa pipeline completo de avaliação.
        Mantém SP e WSP separados durante todo o processo.
        """
        results = []
        processed = 0
        errors = 0

        fasta_files = list(self.balibase_dir.glob("*.tfa"))
        total = len(fasta_files)
        logger.info(f"\nIniciando processamento de {total} arquivos FASTA")

        for i, fasta_file in enumerate(fasta_files, start=1):
            try:
                logger.info(f"\nProcessando {i}/{total}: {fasta_file.name}")

                # 1. Gera alinhamentos
                clustalw_aln = self.run_clustalw(fasta_file)
                muscle_aln = self.run_muscle(fasta_file)

                # 2. Prepara referência
                msf_file = self.reference_dir / f"{fasta_file.stem}.msf"
                if not msf_file.exists():
                    logger.error(f"Referência MSF ausente: {msf_file}")
                    errors += 1
                    continue
                    
                balibase_aln = self.convert_msf_to_aln(msf_file)

                # 3. Avalia cada alinhamento independentemente
                results_dict = {
                    "sequence": fasta_file.stem,
                    "method": ["ClustalW", "MUSCLE", "BAliBASE"],
                    "sp_raw": [],
                    "sp_norm": [],
                    "wsp_raw": [],
                    "wsp_norm": []
                }

                # 4. Avalia cada método
                for method, aln_file in [
                    ("ClustalW", clustalw_aln),
                    ("MUSCLE", muscle_aln),
                    ("BAliBASE", balibase_aln)
                ]:
                    scores = self.evaluate_alignment(aln_file)
                    if not scores:
                        logger.error(f"Falha ao avaliar {method}")
                        continue

                    # Armazena scores separadamente
                    results_dict["sp_raw"].append(scores["sp_raw"])
                    results_dict["sp_norm"].append(scores["sp_norm"])
                    results_dict["wsp_raw"].append(scores["wsp_raw"])
                    results_dict["wsp_norm"].append(scores["wsp_norm"])

                results.append(results_dict)
                processed += 1

            except Exception as e:
                logger.error(f"Erro processando {fasta_file}: {e}")
                errors += 1
                continue

        # Salva resultados em arquivos separados
        if results:
            self.save_results(results)
            logger.info(f"\nPipeline concluído:")
            logger.info(f"  Processados com sucesso: {processed}/{total}")
            logger.info(f"  Erros: {errors}")
            logger.info(f"  SP scores salvos: sp_scores.csv (menor=melhor)")
            logger.info(f"  WSP scores salvos: wsp_scores.csv (maior=melhor)")
        else:
            logger.error("Nenhum resultado gerado")

# =====================================================================
# Execução Principal
# =====================================================================

if __name__ == "__main__":
    balibase_dir = Path("/home/augusto/projects/multiOne/BAliBASE/RV30")
    reference_dir = Path("/home/augusto/projects/multiOne/BAliBASE/RV30")
    results_dir = Path("/home/augusto/projects/multiOne/results")

    pipeline = SequenceAlignmentPipeline(balibase_dir, reference_dir, results_dir)
    pipeline.run_pipeline()