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
#       a) SP = Sum of Pairs (interno)
#          => Calcula, para cada coluna, quantos pares de
#             sequências têm resíduos idênticos (ignorando gaps).
#          => Mostra valores brutos e normalizados [0..1].
#
#       b) WSP = Weighted Sum of Pairs (interno) 
#          => Agora implementado rigorosamente
#             segundo **Gotoh (1994)**, construindo uma árvore
#             (Neighbor Joining) e aplicando a recursão
#             (wsp2_gotoh) para somar as colunas pontuadas.
#          => Mostra valores brutos e normalizados [0..1]
#             para comparação direta com SP.
#
#  3) Ao final, gera um arquivo CSV com 3 linhas por instância:
#       - ClustalW
#       - MUSCLE
#       - BAliBASE (referência)
#     mostrando valores brutos e normalizados das métricas.
#
# ---------------------------------------------------------------------
#  EXTENSÕES E MELHORIAS NA VERSÃO ATUAL:
#
#  1) Normalização do WSP:
#     - Valor bruto: soma ponderada via árvore NJ
#     - Normalizado: WSP_atual/WSP_perfeito, onde:
#       * WSP_atual = score do alinhamento real
#       * WSP_perfeito = score se todas sequências fossem idênticas
#       * Usa mesma topologia/pesos da árvore para comparação justa
#
#  2) Avaliação independente:
#     - Cada alinhamento (.aln) é processado separadamente
#     - BAliBASE MSF -> ALN via seqret também é avaliado por SP/WSP
#     - Permite comparação rigorosa com referência
#
#  3) Logging detalhado:
#     - Mostra progresso de cada etapa
#     - Registra scores brutos e normalizados
#     - Facilita análise e debugging
#
# =========================================================================

LOG_DIR = Path("/dados/home/tesla-dados/multione/results")
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
CLUSTALW_PATH = "/dados/home/tesla-dados/multione/clustalw-2.1/src/clustalw2"
MUSCLE_PATH   = "/dados/home/tesla-dados/multione/muscle-5.3/src/muscle-linux"

# Estrutura para organizar scores brutos e normalizados
class AlignmentScores(NamedTuple):
    sp_raw: float     # SP score bruto (soma direta de matches)
    sp_norm: float    # SP normalizado [0,1] pelo total possível
    wsp_raw: float    # WSP score bruto (soma ponderada via NJ)
    wsp_norm: float   # WSP normalizado [0,1] pelo alinhamento perfeito

# =========================================================================
#                     FUNÇÕES BASE DE PONTUAÇÃO
# =========================================================================

def pairwise_distance(seq1, seq2) -> float:
    """
    Calcula distância evolutiva entre duas sequências.
    Usado tanto para SP quanto para construir árvore NJ.
    
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

def compute_max_score(alignment: MultipleSeqAlignment) -> float:
    """
    Calcula pontuação máxima teórica para normalizar.
    
    Para SP:
    - matches_por_coluna = N*(N-1)/2 onde N = num_seqs
    - total = matches_por_coluna * comprimento
    
    Para WSP:
    - Usa mesma topologia mas sequências idênticas
    - Preserva efeito dos pesos da árvore
    """
    nseqs = len(alignment)
    length = alignment.get_alignment_length()
    
    # Pontuação máxima por coluna (todos os pares matcheando)
    max_col_score = nseqs * (nseqs - 1) / 2
    
    # Total considerando todas as colunas
    return max_col_score * length

def calculate_raw_sp_score(alignment: MultipleSeqAlignment) -> float:
    """
    Calcula SP score sem normalização:
    - Para cada coluna:
      * Compara todos os pares possíveis
      * Soma 1 para cada match (ignorando gaps)
    - Soma global = pontuação final
    """
    nseqs = len(alignment)
    length = alignment.get_alignment_length()
    if nseqs < 2 or length == 0:
        return 0.0

    match_count = 0
    for col in range(length):
        for i in range(nseqs):
            for j in range(i+1, nseqs):
                c1 = alignment[i, col]
                c2 = alignment[j, col]
                if c1 == c2 and c1 != '-':
                    match_count += 1

    return float(match_count)

def calculate_normalized_sp_score(alignment: MultipleSeqAlignment) -> float:
    """
    Normaliza SP score para [0,1]:
    - Calcula score bruto
    - Divide pelo máximo teórico
    - Permite comparação entre alinhamentos diferentes
    """
    raw_score = calculate_raw_sp_score(alignment)
    max_score = compute_max_score(alignment)
    return raw_score / max_score if max_score > 0 else 0.0

# =========================================================================
#                 ESTRUTURAS E FUNÇÕES PARA O WSP REAL (Gotoh)
# =========================================================================

class TreeNode:
    """
    Nó da árvore filogenética para cálculo do WSP.
    
    Cada nó mantém:
    - Identificador único
    - Status (folha/interno)
    - Lista de filhos com pesos dos ramos
    - Sequência (apenas folhas)
    - Perfis P e Q (conforme Gotoh 1994)
    """
    def __init__(self, node_id: str, is_leaf: bool = False):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.children: List[Tuple['TreeNode', float]] = []  # [(nó_filho, comprimento_ramo),...]
        self.sequence: Optional[str] = None
        self.profile_p: Dict[int, Dict[str, float]] = {}  # Perfil P: posição -> {resíduo -> freq}
        self.profile_q: Dict[int, Dict[str, float]] = {}  # Perfil Q: posição -> {resíduo -> freq}

def neighbor_joining_gotoh(dist_matrix: np.ndarray, names: List[str]) -> Optional[TreeNode]:
    """
    Implementa Neighbor Joining com correções para WSP.
    
    Extensões específicas:
    - Controle rigoroso de índices da matriz
    - Preservação de valores na expansão
    - Logging detalhado para diagnóstico
    - Tratamento robusto de erros
    """
    try:
        n_initial = len(names)
        if n_initial < 2:
            return None

        # Cópia para não modificar matriz original
        D = dist_matrix.copy()
        old_size = D.shape[0]
        
        # Inicializa nós e lista de ativos
        nodes = [TreeNode(name, is_leaf=True) for name in names]
        active = list(range(n_initial))
        
        # Iteração principal do NJ
        while len(active) > 1:
            n_active = len(active)
            
            # 1. Soma de linhas para matriz Q
            r = np.array([sum(D[i,j] for j in active if j != i) 
                         for i in active])
            
            # 2. Encontra par mais próximo
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
                
            # 3. Calcula comprimentos dos ramos
            d_ij = D[min_i, min_j]
            if n_active > 2:
                d_i = 0.5 * d_ij + (r[active.index(min_i)] - r[active.index(min_j)]) / (2 * (n_active - 2))
            else:
                d_i = 0.5 * d_ij
            d_j = d_ij - d_i
            
            # 4. Cria novo nó interno
            new_node = TreeNode(f"internal_{len(nodes)}")
            new_node.children.append((nodes[min_i], d_i))
            new_node.children.append((nodes[min_j], d_j))
            nodes.append(new_node)
            
            # 5. Expande matriz de distâncias preservando valores
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
            
            # 6. Atualiza lista de nós ativos
            active.remove(min_i)
            active.remove(min_j)
            active.append(new_idx)
        
        return nodes[-1]  # Retorna raiz
        
    except Exception as e:
        logger.error(f"Erro durante Neighbor Joining: {e}")
        return None

def compute_wsp_gotoh(alignment: MultipleSeqAlignment, 
                     normalize: bool = True, 
                     F: float = 1.15) -> float:
    """
    Implementa WSP conforme Gotoh (1994):
    
    1. Constrói árvore filogenética via NJ
    2. Associa sequências às folhas
    3. Calcula WSP via recursão bottom-up
    4. Opcional: normaliza usando alinhamento perfeito
    
    Parâmetros:
    - alignment: Alinhamento a ser avaliado
    - normalize: Se True, retorna WSP entre [0,1]
    - F: Fator de equalização do artigo original
    
    Retorna:
    - WSP bruto ou normalizado, dependendo do parâmetro
    """
    try:
        if alignment is None or len(alignment) < 2:
            return 0.0
            
        # 1. Constrói matriz de distância
        n_seqs = len(alignment)
        dist_matrix = np.zeros((n_seqs, n_seqs))
        
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                dist = pairwise_distance(alignment[i], alignment[j])
                dist_matrix[i,j] = dist_matrix[j,i] = dist
        
        # 2. Constrói árvore via NJ
        tree = neighbor_joining_gotoh(dist_matrix, [f"seq_{i}" for i in range(n_seqs)])
        if tree is None:
            return 0.0
            
        # 3. Associa sequências às folhas
        stack = [(tree, None)]
        while stack:
            node, parent = stack.pop()
            if node.is_leaf:
                idx = int(node.node_id.split('_')[1])
                node.sequence = str(alignment[idx].seq)
            for child, _ in node.children:
                stack.append((child, node))
                
        # 4. Calcula WSP do alinhamento real
        wsp_raw = wsp2_gotoh(tree)
        if wsp_raw is None:
            return 0.0
            
        # Se não precisar normalizar, retorna valor bruto
        if not normalize:
            return wsp_raw * F
            
        # 5. Calcula WSP do alinhamento perfeito para normalização
        # - Usa mesma árvore/topologia/pesos
        # - Mas todas as sequências são idênticas
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
            
        # 6. Normaliza e aplica fator de equalização
        wsp_normalized = (wsp_raw / wsp_perfect) * F
        
        return wsp_normalized
        
    except Exception as e:
        logger.error(f"Erro no cálculo do WSP: {e}")
        return 0.0

def wsp2_gotoh(node: TreeNode) -> Optional[float]:
    """
    Implementa recursão bottom-up do WSP (Gotoh 1994).
    
    Para cada nó:
    1. Se folha:
       - Converte sequência em perfis P=Q
       - Retorna 0.0 (base da recursão)
       
    2. Se interno:
       - Processa recursivamente os filhos
       - Soma seus scores parciais
       - Combina seus perfis com pesos
       - Adiciona pontuação da combinação
       
    Perfis P e Q guardam frequências dos resíduos
    em cada posição do alinhamento.
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
        
        # Processa filhos recursivamente
        for child, weight in node.children:
            child_score = wsp2_gotoh(child)
            if child_score is None:
                return None
            total_score += child_score
            
            # Combina perfis com pesos
            if not node.profile_p:
                # Primeiro filho inicializa perfis
                node.profile_p = merge_profiles({}, child.profile_p, weight)
                node.profile_q = merge_profiles({}, child.profile_q, weight)
            else:
                # Demais filhos: soma score parcial e atualiza perfis
                partial = weight * dot_product(node.profile_p, child.profile_q)
                total_score += partial
                
                node.profile_p = merge_profiles(node.profile_p, child.profile_p, weight)
                node.profile_q = merge_profiles(node.profile_q, child.profile_q, weight)
                
        return total_score
        
    except Exception as e:
        logger.error(f"Erro na recursão WSP: {e}")
        return None

def sequence_to_profile(seq: str) -> Dict[int, Dict[str, float]]:
    """
    Converte sequência em perfil de frequências.
    
    Para cada posição i da sequência:
    - Cria distribuição sobre resíduos válidos
    - 1.0 para o resíduo observado
    - 0.0 para os demais
    
    Retorna dicionário:
    posição -> {resíduo -> frequência}
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
    Combina dois perfis aplicando peso no segundo.
    
    Para cada posição:
    - Une conjunto de resíduos dos dois perfis
    - Soma frequências ponderadas pelo peso
    
    O peso vem do comprimento do ramo na árvore NJ
    e influencia quanto cada perfil contribui.
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
    
    Para cada posição em comum:
    - Multiplica frequências dos mesmos resíduos
    - Soma todos os produtos
    
    Mede similaridade considerando distribuições
    de resíduos em cada posição.
    """
    score = 0.0
    for pos in set(p1.keys()) & set(p2.keys()):
        for ch in set(p1[pos].keys()) & set(p2[pos].keys()):
            score += p1[pos][ch] * p2[pos][ch]
    return score

# =========================================================================
#                        PIPELINE PRINCIPAL
# =========================================================================

def calculate_scores(alignment: MultipleSeqAlignment) -> Dict[str, float]:
    """
    Função central que calcula todas as métricas para um alinhamento:
    - SP bruto e normalizado
    - WSP bruto e normalizado
    
    Esta função combina os resultados de:
    - calculate_raw_sp_score()
    - calculate_normalized_sp_score()
    - compute_wsp_gotoh() com e sem normalização
    """
    try:
        # SP scores
        sp_raw = calculate_raw_sp_score(alignment)
        sp_norm = calculate_normalized_sp_score(alignment)
        
        # WSP scores (Gotoh)
        wsp_raw = compute_wsp_gotoh(alignment, normalize=False)
        wsp_norm = compute_wsp_gotoh(alignment, normalize=True)
        
        return {
            "sp_raw": sp_raw,
            "sp_norm": sp_norm,
            "wsp_raw": wsp_raw,
            "wsp_norm": wsp_norm
        }
        
    except Exception as e:
        logger.error(f"Erro no cálculo dos scores: {e}")
        return {
            "sp_raw": 0.0,
            "sp_norm": 0.0,
            "wsp_raw": 0.0,
            "wsp_norm": 0.0
        }

class SequenceAlignmentPipeline:
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
        logger.info("Implementa SP bruto/normalizado e WSP (Gotoh 1994) bruto/normalizado")
        logger.info(f"Dir entradas: {self.balibase_dir}")
        logger.info(f"Dir referências: {self.reference_dir}")
        logger.info(f"Dir resultados: {self.results_dir}")

    def run_clustalw(self, input_file: Path) -> Path:
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
        try:
            logger.info(f"\nAvaliando alinhamento: {aln_file}")
            alignment = AlignIO.read(aln_file, "clustal")
            
            # Calcula scores brutos e normalizados
            scores = calculate_scores(alignment)
            
            # Log detalhado
            logger.info("Scores calculados:")
            logger.info(f"  SP  - Bruto: {scores['sp_raw']:.4f}, Normalizado [0,1]: {scores['sp_norm']:.4f}")
            logger.info(f"  WSP - Bruto: {scores['wsp_raw']:.4f}, Normalizado [0,1]: {scores['wsp_norm']:.4f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Erro ao avaliar {aln_file}: {e}")
            return None

    def run_pipeline(self) -> None:
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

                # Avalia cada método
                for method, aln_file in [
                    ("ClustalW", clustalw_aln),
                    ("MUSCLE", muscle_aln),
                    ("BAliBASE", balibase_aln)
                ]:
                    scores = self.evaluate_alignment(aln_file)
                    if not scores:
                        logger.error(f"Falha ao avaliar {method}")
                        continue

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

        # Salva resultados
        if results:
            self._save_results(results)
            logger.info(f"\nPipeline concluído:")
            logger.info(f"  Processados com sucesso: {processed}/{total}")
            logger.info(f"  Erros: {errors}")
        else:
            logger.error("Nenhum resultado gerado")

    def _save_results(self, results: List[Dict]) -> None:
        output_file = self.results_dir / "alignment_scores.csv"
        logger.info(f"\nSalvando resultados: {output_file}")

        fieldnames = [
            "sequence", "method",
            "sp_raw", "sp_norm",
            "wsp_raw", "wsp_norm"
        ]

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
                        "sp_norm": item["sp_norm"][i],
                        "wsp_raw": item["wsp_raw"][i],
                        "wsp_norm": item["wsp_norm"][i]
                    })

        logger.info("Resultados salvos com sucesso")

# =====================================================================
# Execução Principal
# =====================================================================

if __name__ == "__main__":
    balibase_dir = Path("/dados/home/tesla-dados/multione/BAliBASE/RV30")
    reference_dir = Path("/dados/home/tesla-dados/multione/BAliBASE/RV30")
    results_dir = Path("/dados/home/tesla-dados/multione/results")

    pipeline = SequenceAlignmentPipeline(balibase_dir, reference_dir, results_dir)
    pipeline.run_pipeline()