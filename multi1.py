#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Multiple Sequence Alignment Pipeline

Combines:
1. Scientific SP score using PAM250 matrix for improved biological accuracy
2. Original WSP implementation from MultiOne (Gotoh, 1994)
3. Comprehensive pipeline for ClustalW and MUSCLE
4. BAliBASE benchmark processing

Key Features:
- Enhanced SP score using PAM250 substitution matrix
- Preserved original WSP implementation
- Detailed logging and documentation
- Comprehensive evaluation pipeline

Este script tem a finalidade de:
1) Para cada arquivo .tfa (entrada) do BAliBASE:
   - Gerar alinhamentos via ClustalW (em formato .aln)
   - Gerar alinhamentos via MUSCLE (em formato .aln)
   - Converter o arquivo BAliBASE Gold (que está em .msf) 
     para .aln (Clustal)

2) Auto-avaliar cada alinhamento usando:
   a) SP = Sum of Pairs (interno) - Versão aprimorada com PAM250
      => Utiliza matriz PAM250 para pontuação de similaridade
      => Mostra valores brutos e normalizados [0..1]

   b) WSP = Weighted Sum of Pairs (interno)
      => Implementação original segundo Gotoh (1994)
      => Usa árvore Neighbor Joining e recursão wsp2_gotoh
      => Mostra valores brutos e normalizados [0..1]
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime
import math
import numpy as np
import csv

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Configure logging
LOG_DIR = Path("/home/augusto/projects/multiOne/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "alignment_pipeline.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplication
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                            "%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Tool paths - adjust as needed
CLUSTALW_PATH = "/home/augusto/projects/multiOne/clustalw-2.1/src/clustalw2"
MUSCLE_PATH = "/home/augusto/projects/multiOne/muscle-5.3/src/muscle-linux"

class AlignmentScores(NamedTuple):
    """Data structure for organizing alignment scores"""
    sp_raw: float      # SP score (raw) using PAM250
    sp_norm: float     # SP score normalized to [0,1]
    wsp_raw: float     # WSP score (raw) from Gotoh method
    wsp_norm: float    # WSP score normalized to [0,1]

# Base scoring functions
def pairwise_distance(seq1, seq2) -> float:
    """
    Calculate evolutionary distance between two sequences.
    Used for both SP and NJ tree construction.
    
    Returns:
    - 1.0 = totally different sequences
    - 0.0 = identical sequences
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
    Calculate theoretical maximum score for normalization.
    
    For SP:
    - matches_per_column = N*(N-1)/2 where N = num_seqs
    - total = matches_per_column * length
    
    For WSP:
    - Uses same topology but identical sequences
    - Preserves tree weights effect
    """
    nseqs = len(alignment)
    length = alignment.get_alignment_length()
    
    # Maximum score per column (all pairs matching)
    max_col_score = nseqs * (nseqs - 1) / 2
    
    # Total considering all columns
    return max_col_score * length

class SPScoreCalculator:
    """
    Scientific implementation of Sum-of-Pairs (SP) score calculation
    using PAM250 substitution matrix as recommended by Thompson et al. (1999)
    """
    def __init__(self):
        """
        Initialize SP score calculator with PAM250 matrix.
        Pre-calculate matrix extremes for optimization.
        """
        self.matrix = substitution_matrices.load("PAM250")
        self.max_score = max(self.matrix.values())
        self.min_score = min(self.matrix.values())
        self.score_range = self.max_score - self.min_score
        self.logger = logging.getLogger(__name__)

    def compute_sequence_weights(self, alignment: MultipleSeqAlignment) -> np.ndarray:
        """
        Implements CLUSTAL W sequence weighting scheme (Thompson et al., 1994).
        Gives more weight to divergent sequences to reduce sampling bias.
        """
        num_seqs = len(alignment)
        weights = np.ones(num_seqs)
        
        # Calculate pairwise distances using PAM250
        for i in range(num_seqs):
            dists = []
            for j in range(num_seqs):
                if i != j:
                    dist = self._pam_distance(alignment[i], alignment[j])
                    dists.append(dist)
            
            # Weight based on average divergence
            weights[i] = np.mean(dists) if dists else 1.0
            
        # Normalize weights
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
            
        return weights

    def _pam_distance(self, seq1: SeqRecord, seq2: SeqRecord) -> float:
        """
        Calculate evolutionary distance using PAM250 scores.
        Implemented per Dayhoff et al. (1978).
        """
        aligned_positions = 0
        total_score = 0
        
        # Convert sequences to strings for faster access
        seq1_str = str(seq1.seq)
        seq2_str = str(seq2.seq)
        
        for pos in range(len(seq1_str)):
            c1, c2 = seq1_str[pos], seq2_str[pos]
            if c1 != '-' and c2 != '-':
                # Try matrix lookup only once
                score = self.matrix.get((c1, c2), self.matrix.get((c2, c1), 0))
                total_score += score
                aligned_positions += 1
                
        if aligned_positions == 0:
            return 1.0  # Maximum distance for non-aligned sequences
            
        # Normalize to [0,1] using pre-calculated matrix extremes
        normalized_score = (total_score - self.min_score * aligned_positions) / \
                         (self.score_range * aligned_positions)
                         
        return 1.0 - normalized_score

    def compute_sp_score(self, alignment: MultipleSeqAlignment) -> Tuple[float, float]:
            """
            Calculate SP score using PAM250 matrix and sequence weights.
            
            This implementation:
            1. Uses PAM250 for biologically meaningful scoring
            2. Applies sequence weights to reduce redundancy bias
            3. Handles terminal gaps according to BAliBASE guidelines
            4. Returns both raw sum (total matches) and normalized scores
            """
            weights = self.compute_sequence_weights(alignment)
            num_seqs = len(alignment)
            aln_length = alignment.get_alignment_length()
            
            total_matches = 0  # Raw sum of matches
            total_score = 0.0  # Score for normalization
            total_weighted_pairs = 0.0
            
            # Pre-convert sequences to strings
            sequences = [str(seq.seq) for seq in alignment]
            
            for pos in range(aln_length):
                column_score = 0.0
                column_pairs = 0
                
                # Compare all pairs in column
                for i in range(num_seqs-1):
                    for j in range(i+1, num_seqs):
                        res_i = sequences[i][pos]
                        res_j = sequences[j][pos]
                        
                        # Skip terminal gaps per BAliBASE guidelines
                        if self._is_terminal_gap(alignment[i], pos) or \
                        self._is_terminal_gap(alignment[j], pos):
                            continue
                        
                        # Count raw matches
                        if res_i != '-' and res_j != '-':
                            if res_i == res_j:  # Simple match for raw score
                                total_matches += 1
                                
                            # PAM250 score for normalization
                            pair_score = self.matrix.get((res_i, res_j), 
                                                    self.matrix.get((res_j, res_i), 0))
                            weight = weights[i] * weights[j]
                            
                            column_score += pair_score * weight
                            column_pairs += 1
                
                if column_pairs > 0:
                    total_score += column_score
                    total_weighted_pairs += column_pairs
            
            # Return raw match count and normalized PAM250 score
            if total_weighted_pairs == 0:
                return 0.0, 0.0
                
            weighted_score = total_score / total_weighted_pairs
            normalized_score = (weighted_score - self.min_score) / self.score_range
            
            return float(total_matches), normalized_score

    def _is_terminal_gap(self, sequence: SeqRecord, position: int) -> bool:
        """
        Identify terminal gaps according to BAliBASE criteria.
        Terminal gaps often represent genuine biological differences
        rather than alignment errors.
        """
        seq_str = str(sequence.seq)
        return (position == 0 and seq_str[0] == '-') or \
               (position == len(seq_str)-1 and seq_str[-1] == '-')

def calculate_raw_sp_score(alignment: MultipleSeqAlignment) -> float:
    """
    Calculate SP score without normalization:
    - For each column:
      * Compare all possible pairs
      * Add 1 for each match (ignoring gaps)
    - Global sum = final score
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
    Normalize SP score to [0,1]:
    - Calculate raw score
    - Divide by theoretical maximum
    - Enables comparison between different alignments
    """
    raw_score = calculate_raw_sp_score(alignment)
    max_score = compute_max_score(alignment)
    return raw_score / max_score if max_score > 0 else 0.0
# =========================================================================
#                 ESTRUTURAS E FUNÇÕES PARA O WSP (Gotoh)
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
                    q = (n_active - 2) * D[i,j] - r[ai] - r[active.index(j)]
                    if q < min_q:
                        min_q = q
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                return None
                
            # 3. Calcula comprimentos dos ramos
            d_ij = D[min_i, min_j]
            if n_active > 2:
                d_i = 0.5 * d_ij + (r[ai] - r[active.index(min_j)]) / (2 * (n_active - 2))
            else:
                d_i = 0.5 * d_ij
            d_j = d_ij - d_i
            
            # 4. Cria novo nó interno
            new_node = TreeNode(f"internal_{len(nodes)}")
            new_node.children.append((nodes[min_i], d_i))
            new_node.children.append((nodes[min_j], d_j))  # Corrigido aqui
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
    """
    score = 0.0
    for pos in set(p1.keys()) & set(p2.keys()):
        for ch in set(p1[pos].keys()) & set(p2[pos].keys()):
            score += p1[pos][ch] * p2[pos][ch]
    return score

class SequenceAlignmentPipeline:
    """
    Comprehensive pipeline for multiple sequence alignment evaluation.
    Integrates ClustalW, MUSCLE, and BAliBASE benchmark processing
    with both enhanced SP and original WSP scoring.
    """
    def __init__(self, balibase_dir: Path, reference_dir: Path, results_dir: Path):
        """Initialize pipeline with directory configurations and scoring tools"""
        # Directory setup
        self.balibase_dir = balibase_dir
        self.reference_dir = reference_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Tool-specific output directories
        self.clustalw_results_dir = self.results_dir / "clustalw"
        self.muscle_results_dir = self.results_dir / "muscle"
        self.clustalw_results_dir.mkdir(parents=True, exist_ok=True)
        self.muscle_results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scoring calculator
        self.sp_calculator = SPScoreCalculator()

        # Log initialization
        logger.info("\n=== Multiple Sequence Alignment Pipeline Initialized ===")
        logger.info("Using enhanced SP score (PAM250) and original WSP implementation")
        logger.info(f"Input Directory: {self.balibase_dir}")
        logger.info(f"Reference Directory: {self.reference_dir}")
        logger.info(f"Results Directory: {self.results_dir}")
        logger.info("=" * 60)

    def run_clustalw(self, input_file: Path) -> Optional[Path]:
        """Execute ClustalW alignment with core parameters"""
        try:
            output_file = self.clustalw_results_dir / f"{input_file.stem}_clustalw.aln"
            
            # Basic ClustalW command - removed problematic parameters
            cmd = [
                CLUSTALW_PATH,
                "-INFILE=" + str(input_file),
                "-ALIGN",
                "-OUTPUT=CLUSTAL",
                "-OUTFILE=" + str(output_file)
            ]
            
            logger.info(f"[ClustalW] Processing: {input_file.name}")
            result = subprocess.run(cmd, 
                                 check=True, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)

            if not output_file.exists() or output_file.stat().st_size == 0:
                raise RuntimeError("Output file missing or empty")

            logger.info(f"[ClustalW] Successfully generated: {output_file.name}")
            return output_file

        except Exception as e:
            logger.error(f"[ClustalW] Error: {str(e)}")
            return None

    def run_muscle(self, input_file: Path) -> Optional[Path]:
        """Execute MUSCLE alignment with format conversion"""
        try:
            output_fasta = self.muscle_results_dir / f"{input_file.stem}_temp.fa"
            output_file = self.muscle_results_dir / f"{input_file.stem}_muscle.aln"

            logger.info(f"[MUSCLE] Processing: {input_file.name}")
            
            muscle_cmd = [
                str(MUSCLE_PATH),
                "-align", str(input_file),
                "-output", str(output_fasta)
            ]
            
            result = subprocess.run(muscle_cmd,
                                 check=True,
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)

            if not output_fasta.exists():
                raise RuntimeError("MUSCLE alignment failed")

            # Convert FASTA to CLUSTAL format
            logger.info("[MUSCLE] Converting to CLUSTAL format")
            alignments = AlignIO.read(output_fasta, "fasta")
            AlignIO.write(alignments, output_file, "clustal")

            # Clean up temporary file
            output_fasta.unlink()

            if not output_file.exists():
                raise RuntimeError("Format conversion failed")

            logger.info(f"[MUSCLE] Successfully generated: {output_file.name}")
            return output_file

        except Exception as e:
            logger.error(f"[MUSCLE] Error: {str(e)}")
            return None

    def convert_msf_to_aln(self, msf_file: Path) -> Optional[Path]:
        """Convert BAliBASE MSF reference to CLUSTAL format"""
        try:
            aln_file = msf_file.with_suffix(".aln")
            
            # Use existing conversion if available
            if aln_file.exists() and aln_file.stat().st_size > 0:
                logger.info(f"[BAliBASE] Using existing conversion: {aln_file.name}")
                return aln_file

            logger.info(f"[BAliBASE] Converting: {msf_file.name}")
            
            cmd = [
                "seqret",
                "-sequence", str(msf_file),
                "-outseq", str(aln_file),
                "-osformat2", "clustal"
            ]
            
            result = subprocess.run(cmd,
                                 check=True,
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)

            if not aln_file.exists():
                raise RuntimeError("Conversion failed")

            logger.info(f"[BAliBASE] Successfully converted to: {aln_file.name}")
            return aln_file

        except Exception as e:
            logger.error(f"[BAliBASE] Conversion error: {str(e)}")
            return None

    def evaluate_alignment(self, aln_file: Path) -> Optional[Dict[str, float]]:
        """
        Evaluate alignment using both enhanced SP and original WSP scores.
        Returns dictionary with raw and normalized scores.
        """
        try:
            logger.info(f"\nEvaluating alignment: {aln_file}")
            
            alignment = AlignIO.read(aln_file, "clustal")
            
            # Calculate SP scores using PAM250
            sp_raw, sp_norm = self.sp_calculator.compute_sp_score(alignment)
            
            # Calculate WSP scores usando implementação original
            wsp_raw = compute_wsp_gotoh(alignment, normalize=False)
            wsp_norm = compute_wsp_gotoh(alignment, normalize=True)
            
            scores = {
                "sp_raw": sp_raw,
                "sp_norm": sp_norm,
                "wsp_raw": wsp_raw,
                "wsp_norm": wsp_norm
            }
            
            logger.info("Scores calculated:")
            logger.info(f"  SP Score (PAM250):")
            logger.info(f"    Raw: {sp_raw:.4f}")
            logger.info(f"    Normalized [0,1]: {sp_norm:.4f}")
            logger.info(f"  WSP Score (Gotoh):")
            logger.info(f"    Raw: {wsp_raw:.4f}")
            logger.info(f"    Normalized [0,1]: {wsp_norm:.4f}")
            
            return scores

        except Exception as e:
            logger.error(f"Evaluation error for {aln_file}: {str(e)}")
            return None

    def _generate_alignment_statistics(self, aln_file: Path, method: str) -> None:
        """Generate detailed statistics for each alignment method"""
        try:
            alignment = AlignIO.read(aln_file, "clustal")
            
            num_sequences = len(alignment)
            alignment_length = alignment.get_alignment_length()
            
            # Analyze gap distribution
            gap_counts = []
            for record in alignment:
                gaps = str(record.seq).count('-')
                gap_percentage = (gaps / alignment_length) * 100
                gap_counts.append(gap_percentage)
            
            avg_gap_percentage = np.mean(gap_counts)
            
            logger.info(f"\n{method} Alignment Statistics:")
            logger.info(f"Sequences: {num_sequences}")
            logger.info(f"Alignment Length: {alignment_length}")
            logger.info(f"Average Gap Percentage: {avg_gap_percentage:.2f}%")
            
        except Exception as e:
            logger.error(f"Error generating statistics for {method}: {str(e)}")

    def _calculate_comparative_metrics(self, sequence_results: Dict) -> None:
        """Calculate comparative metrics between different methods"""
        try:
            methods = sequence_results["methods"]
            sp_scores = sequence_results["sp_norm"]
            wsp_scores = sequence_results["wsp_norm"]
            
            logger.info("\nComparative Analysis:")
            for i, method1 in enumerate(methods[:-1]):
                for method2 in methods[i+1:]:
                    j = methods.index(method2)
                    sp_diff = abs(sp_scores[i] - sp_scores[j])
                    wsp_diff = abs(wsp_scores[i] - wsp_scores[j])
                    
                    logger.info(f"\n{method1} vs {method2}:")
                    logger.info(f"SP Score Difference: {sp_diff:.4f}")
                    logger.info(f"WSP Score Difference: {wsp_diff:.4f}")
                    
        except Exception as e:
            logger.error(f"Error calculating comparative metrics: {str(e)}")

    def _save_results(self, results: List[Dict]) -> None:
        """Save comprehensive results to CSV format"""
        try:
            output_file = self.results_dir / "alignment_scores.csv"
            logger.info(f"\nSaving results to: {output_file}")

            fieldnames = [
                "sequence", "method",
                "sp_raw", "sp_norm",
                "wsp_raw", "wsp_norm"
            ]

            with open(output_file, "w", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for item in results:
                    seq_id = item["sequence_id"]
                    for i, method in enumerate(item["methods"]):
                        writer.writerow({
                            "sequence": seq_id,
                            "method": method,
                            "sp_raw": item["sp_raw"][i],
                            "sp_norm": item["sp_norm"][i],
                            "wsp_raw": item["wsp_raw"][i],
                            "wsp_norm": item["wsp_norm"][i]
                        })

            logger.info("Results saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def _generate_summary_report(self, results: List[Dict], 
                               processed: int, errors: int, total: int) -> None:
        """Generate comprehensive summary report"""
        try:
            report_file = self.results_dir / "analysis_report.txt"
            
            with open(report_file, "w") as f:
                f.write("Multiple Sequence Alignment Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Processing Statistics:\n")
                f.write(f"Total sequences: {total}\n")
                f.write(f"Successfully processed: {processed}\n")
                f.write(f"Errors: {errors}\n\n")
                
                f.write("Method Comparison:\n")
                for method in ["ClustalW", "MUSCLE", "BAliBASE"]:
                    method_scores = [item for item in results 
                                   if method in item["methods"]]
                    
                    if method_scores:
                        sp_scores = [item["sp_norm"][item["methods"].index(method)]
                                   for item in method_scores]
                        wsp_scores = [item["wsp_norm"][item["methods"].index(method)]
                                    for item in method_scores]
                        
                        f.write(f"\n{method}:\n")
                        f.write(f"Average SP Score (PAM250): {np.mean(sp_scores):.4f}\n")
                        f.write(f"Average WSP Score (Gotoh): {np.mean(wsp_scores):.4f}\n")
                        
            logger.info(f"Summary report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")

    def run_pipeline(self) -> None:
        """Execute the complete alignment and evaluation pipeline"""
        results = []
        processed = 0
        errors = 0

        # Collect and validate input files
        fasta_files = list(self.balibase_dir.glob("*.tfa"))
        total = len(fasta_files)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting pipeline processing for {total} sequences")
        logger.info(f"{'='*60}")

        for i, fasta_file in enumerate(fasta_files, start=1):
            try:
                logger.info(f"\nProcessing sequence {i}/{total}")
                logger.info(f"File: {fasta_file.name}")
                logger.info("-" * 40)

                start_time = datetime.now()

                # 1. Generate alignments with different tools
                alignments = {}
                
                if clustalw_aln := self.run_clustalw(fasta_file):
                    alignments["ClustalW"] = clustalw_aln
                
                if muscle_aln := self.run_muscle(fasta_file):
                    alignments["MUSCLE"] = muscle_aln
                
                # Process BAliBASE reference
                msf_file = self.reference_dir / f"{fasta_file.stem}.msf"
                if msf_file.exists():
                    if balibase_aln := self.convert_msf_to_aln(msf_file):
                        alignments["BAliBASE"] = balibase_aln
                else:
                    logger.warning(f"Reference MSF not found: {msf_file}")

                # 2. Evaluate each alignment
                sequence_results = {
                    "sequence_id": fasta_file.stem,
                    "methods": [],
                    "sp_raw": [],
                    "sp_norm": [],
                    "wsp_raw": [],
                    "wsp_norm": []
                }

                for method, aln_file in alignments.items():
                    logger.info(f"\nEvaluating {method} alignment")
                    
                    if scores := self.evaluate_alignment(aln_file):
                        sequence_results["methods"].append(method)
                        sequence_results["sp_raw"].append(scores["sp_raw"])
                        sequence_results["sp_norm"].append(scores["sp_norm"])
                        sequence_results["wsp_raw"].append(scores["wsp_raw"])
                        sequence_results["wsp_norm"].append(scores["wsp_norm"])
                        
                        self._generate_alignment_statistics(aln_file, method)

                # 3. Calculate comparative metrics
                if len(sequence_results["methods"]) > 1:
                    self._calculate_comparative_metrics(sequence_results)

                results.append(sequence_results)
                processed += 1

                duration = datetime.now() - start_time
                logger.info(f"\nSequence completed in {duration}")

            except Exception as e:
                logger.error(f"Error processing {fasta_file}: {str(e)}")
                errors += 1
                continue

        # Generate final results and reports
        if results:
            self._save_results(results)
            self._generate_summary_report(results, processed, errors, total)
        else:
            logger.error("No results generated")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    try:
        # Configure directories
        balibase_dir = Path("/home/augusto/projects/multiOne/BAliBASE/RV30")
        reference_dir = Path("/home/augusto/projects/multiOne/BAliBASE/RV30")
        results_dir = Path("/home/augusto/projects/multiOne/results")

        # Initialize and run pipeline
        pipeline = SequenceAlignmentPipeline(
            balibase_dir=balibase_dir,
            reference_dir=reference_dir,
            results_dir=results_dir
        )
        
        pipeline.run_pipeline()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)
