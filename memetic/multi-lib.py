#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GECCO 2025 - Multiple Sequence Alignment Project
------------------------------------------------
Título: "Melhorando a Avaliação de Alinhamentos Múltiplos com
       SP Score Otimizado (PAM250 + Cache) e WSP (Gotoh)"

Objetivos:
1) Ler sequências de entrada do BAliBASE (.tfa)
2) Gerar alinhamentos (ClustalW e MUSCLE)
3) Converter .msf do BAliBASE em .aln
4) Avaliar cada alinhamento com:
   (a) SP Score - PAM250 com cache (sp_raw + sp_norm [0..1])
   (b) WSP Score - Gotoh (1994) (wsp_raw + wsp_norm [0..1])

Justificativa Científica:
- PAM250 (Dayhoff et al., 1978) é indicada para sequências evolutivamente distantes.
- WSP (Gotoh, 1994) leva em conta filogenia via Neighbor Joining e recursão.
- Uso de cache e filtragem de colunas acelera o SP Score.
- Gaps terminais não penalizam (Thompson et al., 1999).
- Sequências muito semelhantes recebem menos peso (ClustalW weighting).

Matemática (resumida):
- sp_raw = Σ (par_score × peso_i × peso_j), somando sobre colunas sem gap terminal
- sp_norm = (sp_real_sum - sp_min_possible)/(sp_max_possible - sp_min_possible)
- wsp_raw = soma de perfis via wsp2_gotoh
- Normalização do WSP => wsp_norm = (wsp_raw / wsp_perfect)*F

------------------------------------------------
Pipeline a seguir:
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

# ------------------------------------------
# LOGGING CONFIG
# ------------------------------------------
LOG_DIR = Path("/home/augusto/projects/multiOne/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "alignment_pipeline.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove old handlers to avoid duplication
for h in list(logger.handlers):
    logger.removeHandler(h)

fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
fh  = logging.FileHandler(LOG_FILE, mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)

ch  = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)

logger.addHandler(fh)
logger.addHandler(ch)

# ------------------------------------------
# External Tools
# ------------------------------------------
CLUSTALW_PATH = "/home/augusto/projects/multiOne/clustalw-2.1/src/clustalw2"
MUSCLE_PATH   = "/home/augusto/projects/multiOne/muscle-5.3/src/muscle-linux"

class AlignmentScores(NamedTuple):
    """
    ESTRUTURA DE DADOS PARA ARMAZENAR SCORES
    sp_raw   : SP Score bruto (PAM250)
    sp_norm  : SP normalizado [0..1]
    wsp_raw  : Weighted Sum of Pairs (Gotoh) bruto
    wsp_norm : Weighted Sum of Pairs (Gotoh) normalizado [0..1]
    """
    sp_raw : float
    sp_norm: float
    wsp_raw: float
    wsp_norm: float

# ------------------------------------------
# PAIRWISE DISTANCE (Para WSP, NJ, etc.)
# ------------------------------------------
def pairwise_distance(seq1, seq2) -> float:
    """
    Distância evolutiva simples: 1.0 - (matches/total), ignorando gaps
    Retorna 1.0 se completamente diferentes (ou sem posições comparáveis).
    """
    if len(seq1) != len(seq2):
        return 1.0
    matches = 0
    total = 0
    for c1, c2 in zip(seq1, seq2):
        if c1 != '-' and c2 != '-':
            total += 1
            if c1 == c2:
                matches += 1
    if total == 0:
        return 1.0
    return 1.0 - (matches / total)

# ------------------------------------------
# SP SCORE (OTIMIZADO) COM PAM250
# ------------------------------------------
class SPScoreCalculator:
    """
    Implementa SP Score (Sum-of-Pairs):
    - PAM250 (Dayhoff et al. 1978)
    - Pesos de sequência (ClustalW)
    - Filtragem de colunas (evita gap terminal)
    - Retorna (sp_raw, sp_norm)
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Carrega PAM250
        self.matrix = substitution_matrices.load("PAM250")
        self.pam_min = min(self.matrix.values())
        self.pam_max = max(self.matrix.values())
        # Cache { (res_i, res_j): score }
        self.score_cache = {}
        for (a1,a2), val in self.matrix.items():
            self.score_cache[(a1,a2)] = val
            self.score_cache[(a2,a1)] = val

    def compute_sp_score(self, alignment: MultipleSeqAlignment)-> Tuple[float,float]:
        """
        Retorna (sp_raw, sp_norm):
         - sp_raw: soma ponderada dos matches exatos (c1==c2)
         - sp_norm: normalizado em [0..1] (via min_possible, max_possible)
        """
        # 1) Calcula pesos de sequência
        weights = self._compute_clustalw_weights(alignment)
        n= len(alignment)
        length= alignment.get_alignment_length()

        # Acumuladores
        sp_raw_sum  = 0.0  # soma de matches ponderados
        sp_min_sum  = 0.0
        sp_max_sum  = 0.0
        sp_real_sum = 0.0
        total_pairs = 0

        seqs_str= [str(rec.seq) for rec in alignment]

        for col in range(length):
            # Filtra idxs válidos nessa coluna (sem gap terminal e sem '-')
            valid_idxs= []
            for i in range(n):
                if not self._is_terminal_gap(alignment[i], col):
                    if seqs_str[i][col] != '-':
                        valid_idxs.append(i)

            if len(valid_idxs)<2:
                continue

            col_real=0.0
            col_min =0.0
            col_max =0.0
            col_pairs=0

            # par (x, y) em valid_idxs
            for idx_x in range(len(valid_idxs)-1):
                x= valid_idxs[idx_x]
                for idx_y in range(idx_x+1, len(valid_idxs)):
                    y= valid_idxs[idx_y]
                    c1= seqs_str[x][col]
                    c2= seqs_str[y][col]

                    pair_score= self.score_cache.get((c1,c2), 0)
                    w = weights[x]* weights[y]

                    col_real+= pair_score* w
                    col_min += self.pam_min* w
                    col_max += self.pam_max* w

                    # Se for match literal, soma no sp_raw
                    if c1 == c2:
                        sp_raw_sum+= w

                    col_pairs+=1

            if col_pairs>0:
                sp_real_sum += col_real
                sp_min_sum  += col_min
                sp_max_sum  += col_max
                total_pairs += col_pairs

        if total_pairs==0:
            return (0.0,0.0)

        denom= sp_max_sum - sp_min_sum
        if denom==0:
            return (sp_raw_sum, 0.0)

        sp_norm= (sp_real_sum - sp_min_sum)/ denom

        self.logger.info(
            f"[SPScore] raw={sp_raw_sum:.3f} real={sp_real_sum:.3f} "
            f"min={sp_min_sum:.3f} max={sp_max_sum:.3f} norm={sp_norm:.4f}"
        )
        return (sp_raw_sum, sp_norm)

    def _compute_clustalw_weights(self, alignment: MultipleSeqAlignment)-> np.ndarray:
        """
        Pesos de sequência => dist ~ 1 - normalização P( seq_i, seq_j ) => media => weighting
        """
        n= len(alignment)
        weights= np.ones(n, dtype=float)
        for i in range(n):
            dists= []
            si= str(alignment[i].seq)
            for j in range(n):
                if i!= j:
                    sj= str(alignment[j].seq)
                    d= self._pam_distance(si, sj)
                    dists.append(d)
            weights[i]= np.mean(dists) if dists else 1.0

        s= np.sum(weights)
        if s>0:
            weights/= s
        return weights

    def _pam_distance(self, seq_i: str, seq_j: str)-> float:
        """
        Distância [0..1], invertendo pontuação PAM250
        """
        aligned_positions=0
        total_score=0
        for c1, c2 in zip(seq_i, seq_j):
            if c1!='-' and c2!='-':
                sc= self.score_cache.get((c1,c2), 0)
                total_score+= sc
                aligned_positions+=1
        if aligned_positions==0:
            return 1.0
        rng= self.pam_max- self.pam_min
        if rng==0:
            return 0.0
        normalized_score= (total_score - self.pam_min*aligned_positions)/(rng*aligned_positions)
        return 1.0- normalized_score

    def _is_terminal_gap(self, seq_record:SeqRecord, pos:int)->bool:
        """
        BAliBASE => gap terminal => skip
        """
        seq_str= str(seq_record.seq)
        if pos==0 and seq_str[0]=='-':
            return True
        if pos== len(seq_str)-1 and seq_str[-1]=='-':
            return True
        return False

# ------------------------------------------
#   WSP (Gotoh 1994)
# ------------------------------------------
class TreeNode:
    """
    Nó filogenético para WSP (Gotoh).
    """
    def __init__(self, node_id:str, is_leaf:bool=False):
        self.node_id= node_id
        self.is_leaf= is_leaf
        self.children: List[Tuple['TreeNode',float]] = []
        self.sequence:Optional[str]= None
        self.profile_p: Dict[int,Dict[str,float]] = {}
        self.profile_q: Dict[int,Dict[str,float]] = {}

def neighbor_joining_gotoh(dist_matrix: np.ndarray, names:List[str])-> Optional[TreeNode]:
    """
    NJ => correções para WSP
    IMPORTANTE: correção de index => r[active.index(i)] e r[active.index(j)]
    """
    try:
        n_initial= len(names)
        if n_initial<2:
            return None

        D= dist_matrix.copy()
        old_size= D.shape[0]
        nodes= [TreeNode(name, True) for name in names]
        active= list(range(n_initial))

        while len(active)>1:
            n_act= len(active)
            # array r
            r = np.array([sum(D[i, k] for k in active if k!= i) for i in active])

            min_q= float('inf')
            min_i= min_j= -1

            # loop
            for ai, i in enumerate(active[:-1]):
                for j in active[ai+1:]:
                    # Corrige => idx_i= active.index(i), idx_j= active.index(j)
                    idx_i= active.index(i)
                    idx_j= active.index(j)
                    q= (n_act-2)*D[i,j] - r[idx_i] - r[idx_j]
                    if q< min_q:
                        min_q= q
                        min_i, min_j= i, j

            if min_i<0 or min_j<0:
                return None

            d_ij= D[min_i, min_j]
            idx_i= active.index(min_i)
            idx_j= active.index(min_j)
            if n_act>2:
                d_i= 0.5*d_ij + (r[idx_i] - r[idx_j])/(2*(n_act-2))
            else:
                d_i= 0.5*d_ij
            d_j= d_ij- d_i

            new_node= TreeNode(f"internal_{len(nodes)}", False)
            new_node.children.append((nodes[min_i], d_i))
            new_node.children.append((nodes[min_j], d_j))
            nodes.append(new_node)

            new_idx= len(nodes)-1
            if new_idx>= old_size:
                D_new= np.zeros((old_size+1, old_size+1))
                D_new[:old_size,:old_size]= D
                D= D_new
                old_size+=1

            for k in active:
                if k not in (min_i,min_j):
                    dist_k= 0.5*(D[min_i, k]+ D[min_j, k]- D[min_i, min_j])
                    D[new_idx, k]= dist_k
                    D[k, new_idx]= dist_k

            active.remove(min_i)
            active.remove(min_j)
            active.append(new_idx)

        return nodes[-1]
    except Exception as e:
        logger.error(f"Erro NJ: {e}")
        return None

def wsp2_gotoh(node: TreeNode)-> Optional[float]:
    """
    Recursão bottom-up do WSP
    """
    try:
        if node is None:
            return None
        if node.is_leaf:
            if node.sequence is None:
                return None
            node.profile_p= sequence_to_profile(node.sequence)
            node.profile_q= sequence_to_profile(node.sequence)
            return 0.0

        total_score=0.0
        for child, weight in node.children:
            child_score= wsp2_gotoh(child)
            if child_score is None:
                return None
            total_score+= child_score

            if not node.profile_p:
                node.profile_p= merge_profiles({}, child.profile_p, weight)
                node.profile_q= merge_profiles({}, child.profile_q, weight)
            else:
                partial= weight * dot_product(node.profile_p, child.profile_q)
                total_score+= partial
                node.profile_p= merge_profiles(node.profile_p, child.profile_p, weight)
                node.profile_q= merge_profiles(node.profile_q, child.profile_q, weight)
        return total_score
    except Exception as e:
        logger.error(f"Erro recursion WSP: {e}")
        return None

def compute_wsp_gotoh(alignment:MultipleSeqAlignment, normalize:bool=True, F:float=1.15)-> float:
    """
    WSP => neighbor_joining_gotoh => wsp2_gotoh => normalização
    """
    try:
        if not alignment or len(alignment)<2:
            return 0.0
        n= len(alignment)
        D= np.zeros((n,n), dtype=float)

        # dist matrix
        for i in range(n):
            for j in range(i+1,n):
                dist= pairwise_distance(alignment[i], alignment[j])
                D[i,j]= dist
                D[j,i]= dist

        # NJ
        tree= neighbor_joining_gotoh(D, [f"seq_{i}" for i in range(n)])
        if not tree:
            return 0.0

        # Associa seqs
        stk= [(tree,None)]
        while stk:
            nd, pr= stk.pop()
            if nd.is_leaf:
                idx= int(nd.node_id.split('_')[1])
                nd.sequence= str(alignment[idx].seq)
            for ch, _ in nd.children:
                stk.append((ch, nd))

        wsp_raw= wsp2_gotoh(tree)
        if wsp_raw is None:
            return 0.0
        if not normalize:
            return wsp_raw*F

        # Alinhamento perfeito => "A"
        perfect_seq= "A"* alignment.get_alignment_length()
        perfect_tree= neighbor_joining_gotoh(D, [f"seq_{i}" for i in range(n)])
        if not perfect_tree:
            return wsp_raw*F

        stk= [(perfect_tree,None)]
        while stk:
            nd, pr= stk.pop()
            if nd.is_leaf:
                nd.sequence= perfect_seq
            for ch,_ in nd.children:
                stk.append((ch,nd))

        wsp_perfect= wsp2_gotoh(perfect_tree)
        if not wsp_perfect or wsp_perfect==0.0:
            return wsp_raw*F
        return (wsp_raw/ wsp_perfect)*F
    except Exception as e:
        logger.error(f"Erro WSP: {e}")
        return 0.0

# Perfis
def sequence_to_profile(seq:str)-> Dict[int,Dict[str,float]]:
    valid_chars= set("ACDEFGHIKLMNPQRSTVWY-")
    pf={}
    for i, c in enumerate(seq):
        freqs= {ch:0.0 for ch in valid_chars}
        if c in valid_chars:
            freqs[c]= 1.0
        pf[i]= freqs
    return pf

def merge_profiles(p1:Dict[int,Dict[str,float]], p2:Dict[int,Dict[str,float]], w:float)-> Dict[int,Dict[str,float]]:
    res={}
    all_pos= set(p1.keys())| set(p2.keys())
    valid_chars= set("ACDEFGHIKLMNPQRSTVWY-")
    for pos in all_pos:
        freqs= {ch:0.0 for ch in valid_chars}
        for ch in valid_chars:
            v1= p1.get(pos,{}).get(ch,0.0)
            v2= p2.get(pos,{}).get(ch,0.0)
            freqs[ch]= w*(v1+v2)
        res[pos]= freqs
    return res

def dot_product(p1:Dict[int,Dict[str,float]], p2:Dict[int,Dict[str,float]])-> float:
    s=0.0
    for pos in set(p1.keys()) & set(p2.keys()):
        for ch in set(p1[pos].keys()) & set(p2[pos].keys()):
            s+= p1[pos][ch]* p2[pos][ch]
    return s

# -------------------------------------------
# PIPELINE PRINCIPAL
# -------------------------------------------
class SequenceAlignmentPipeline:
    """
    Pipeline final que:
     1) Identifica .tfa no BAliBASE
     2) Gera ClustalW e MUSCLE
     3) Converte BAliBASE .msf => .aln
     4) Avalia SP Score (otimizado) e WSP
     5) Salva CSV, gera relatório e comparativos
    """
    def __init__(self, balibase_dir:Path, reference_dir:Path, results_dir:Path):
        self.balibase_dir   = balibase_dir
        self.reference_dir  = reference_dir
        self.results_dir    = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.clustalw_results_dir= self.results_dir/"clustalw"
        self.muscle_results_dir=   self.results_dir/"muscle"
        self.clustalw_results_dir.mkdir(parents=True, exist_ok=True)
        self.muscle_results_dir.mkdir(parents=True, exist_ok=True)

        # SPCalculator
        self.sp_calculator= SPScoreCalculator()

        logger.info("\n=== Multiple Sequence Alignment Pipeline Initialized ===")
        logger.info("Using optimized SP (PAM250) and original WSP (Gotoh)")
        logger.info(f"Input Dir: {self.balibase_dir}")
        logger.info(f"Reference Dir: {self.reference_dir}")
        logger.info(f"Results Dir: {self.results_dir}")
        logger.info("="*60)

    def run_clustalw(self, input_file:Path)-> Optional[Path]:
        """
        Executa ClustalW => .aln
        """
        try:
            output_file= self.clustalw_results_dir/ f"{input_file.stem}_clustalw.aln"
            cmd= [
                CLUSTALW_PATH,
                "-INFILE="+ str(input_file),
                "-ALIGN",
                "-OUTPUT=CLUSTAL",
                "-OUTFILE="+ str(output_file)
            ]
            logger.info(f"[ClustalW] Processing: {input_file.name}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if not output_file.exists() or output_file.stat().st_size==0:
                raise RuntimeError("ClustalW: missing or empty output")

            logger.info(f"[ClustalW] Successfully generated: {output_file.name}")
            return output_file
        except Exception as e:
            logger.error(f"[ClustalW] Error: {e}")
            return None

    def run_muscle(self, input_file:Path)-> Optional[Path]:
        """
        Executa MUSCLE => converte => .aln
        """
        try:
            output_fasta= self.muscle_results_dir/ f"{input_file.stem}_temp.fa"
            output_file=  self.muscle_results_dir/ f"{input_file.stem}_muscle.aln"

            logger.info(f"[MUSCLE] Processing: {input_file.name}")
            muscle_cmd= [
                str(MUSCLE_PATH),
                "-align", str(input_file),
                "-output", str(output_fasta)
            ]
            subprocess.run(muscle_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if not output_fasta.exists():
                raise RuntimeError("MUSCLE alignment failed")

            # Converte => clustal
            logger.info("[MUSCLE] Converting to CLUSTAL format")
            alignments= AlignIO.read(output_fasta, "fasta")
            AlignIO.write(alignments, output_file, "clustal")

            output_fasta.unlink()
            if not output_file.exists() or output_file.stat().st_size==0:
                raise RuntimeError("Format conversion failed")

            logger.info(f"[MUSCLE] Successfully generated: {output_file.name}")
            return output_file

        except Exception as e:
            logger.error(f"[MUSCLE] Error: {e}")
            return None

    def convert_msf_to_aln(self, msf_file:Path)-> Optional[Path]:
        """
        Converte BAliBASE .msf => .aln via seqret
        """
        try:
            aln_file= msf_file.with_suffix(".aln")
            if aln_file.exists() and aln_file.stat().st_size>0:
                logger.info(f"[BAliBASE] Using existing conversion: {aln_file.name}")
                return aln_file

            logger.info(f"[BAliBASE] Converting: {msf_file.name}")
            cmd= [
                "seqret",
                "-sequence", str(msf_file),
                "-outseq",  str(aln_file),
                "-osformat2","clustal"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if not aln_file.exists():
                raise RuntimeError("Conversion failed")

            logger.info(f"[BAliBASE] Successfully converted to: {aln_file.name}")
            return aln_file
        except Exception as e:
            logger.error(f"[BAliBASE] Conversion error: {e}")
            return None

    def evaluate_alignment(self, aln_file:Path)-> Optional[Dict[str,float]]:
        """
        Faz a avaliação:
         sp_raw, sp_norm (PAM250)
         wsp_raw, wsp_norm (Gotoh)
        """
        try:
            logger.info(f"\nEvaluating alignment: {aln_file}")
            alignment= AlignIO.read(aln_file, "clustal")

            # SP
            sp_raw, sp_norm= self.sp_calculator.compute_sp_score(alignment)

            # WSP
            wsp_raw= compute_wsp_gotoh(alignment, normalize=False)
            wsp_norm= compute_wsp_gotoh(alignment, normalize=True)

            scores= {
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
            logger.error(f"Evaluation error for {aln_file}: {e}")
            return None

    def _generate_alignment_statistics(self, aln_file:Path, method:str)-> None:
        """
        Coletar estatísticas: gap distribution, etc.
        """
        try:
            alignment= AlignIO.read(aln_file, "clustal")
            num_seq= len(alignment)
            aln_len= alignment.get_alignment_length()

            gap_counts= []
            for rec in alignment:
                gaps= str(rec.seq).count('-')
                gap_pct= (gaps/aln_len)*100
                gap_counts.append(gap_pct)

            avg_gap= np.mean(gap_counts)

            logger.info(f"\n{method} Alignment Statistics:")
            logger.info(f"Sequences: {num_seq}")
            logger.info(f"Alignment Length: {aln_len}")
            logger.info(f"Average Gap Percentage: {avg_gap:.2f}%")
        except Exception as e:
            logger.error(f"Error generating statistics for {method}: {e}")

    def _calculate_comparative_metrics(self, seq_results:Dict)-> None:
        """
        "Comparative Analysis across methods:"
        exibe diferença entre sp_norm e wsp_norm
        """
        try:
            meths= seq_results["methods"]
            if len(meths)<2:
                return

            logger.info("\nComparative Analysis across methods:")
            spv= seq_results["sp_norm"]
            wspv= seq_results["wsp_norm"]
            for i, m1 in enumerate(meths[:-1]):
                for j, m2 in enumerate(meths[i+1:], start=i+1):
                    sp_diff= abs(spv[i]- spv[j])
                    wsp_diff= abs(wspv[i]- wspv[j])
                    logger.info(f"\n{m1} vs {m2}:")
                    logger.info(f"  SP Score Difference:  {sp_diff:.4f}")
                    logger.info(f"  WSP Score Difference: {wsp_diff:.4f}")
        except Exception as e:
            logger.error(f"Error in comparative metrics: {e}")

    def _save_results(self, results:List[Dict])-> None:
        """
        Salva sp_raw/sp_norm + wsp_raw/wsp_norm em CSV
        """
        try:
            output_file= self.results_dir/ "alignment_scores.csv"
            logger.info(f"\nSaving results to: {output_file}")

            flds= ["sequence","method","sp_raw","sp_norm","wsp_raw","wsp_norm"]
            with open(output_file, "w", newline='') as csvf:
                writer= csv.DictWriter(csvf, fieldnames=flds)
                writer.writeheader()
                for item in results:
                    seq_id= item["sequence_id"]
                    for i, meth in enumerate(item["methods"]):
                        writer.writerow({
                            "sequence": seq_id,
                            "method":   meth,
                            "sp_raw":   item["sp_raw"][i],
                            "sp_norm":  item["sp_norm"][i],
                            "wsp_raw":  item["wsp_raw"][i],
                            "wsp_norm": item["wsp_norm"][i]
                        })
            logger.info("Results saved successfully")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def _generate_summary_report(self, results:List[Dict], processed:int, errors:int, total:int)-> None:
        """
        Gera relatório textual => com média de SP e WSP
        """
        try:
            rep_file= self.results_dir/ "analysis_report.txt"
            with open(rep_file, "w") as f:
                f.write("Multiple Sequence Alignment Analysis Report\n")
                f.write("="*50+"\n\n")

                f.write("Processing Statistics:\n")
                f.write(f"Total sequences: {total}\n")
                f.write(f"Successfully processed: {processed}\n")
                f.write(f"Errors: {errors}\n\n")

                f.write("Method Comparison:\n")
                for method in ["ClustalW","MUSCLE","BAliBASE"]:
                    meth_scores= [r for r in results if method in r["methods"]]
                    if meth_scores:
                        sp_vals=[]
                        wsp_vals=[]
                        for sc in meth_scores:
                            idx= sc["methods"].index(method)
                            sp_vals.append(sc["sp_norm"][idx])
                            wsp_vals.append(sc["wsp_norm"][idx])
                        f.write(f"\n{method}:\n")
                        f.write(f"Average SP Score (PAM250): {np.mean(sp_vals):.4f}\n")
                        f.write(f"Average WSP Score (Gotoh):  {np.mean(wsp_vals):.4f}\n")

            logger.info(f"Summary report generated: {rep_file}")
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

    def run_pipeline(self)-> None:
        """
        1) Identifica .tfa
        2) Gera ClustalW, MUSCLE, BAliBASE
        3) Avalia => sp/wsp
        4) Gera CSV + relatório
        """
        results=[]
        processed=0
        errors=0
        fasta_files= list(self.balibase_dir.glob("*.tfa"))
        total= len(fasta_files)

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting pipeline processing for {total} sequences")
        logger.info(f"{'='*60}")

        for i, f in enumerate(fasta_files, start=1):
            try:
                logger.info(f"\nProcessing sequence {i}/{total}")
                logger.info(f"File: {f.name}")
                logger.info("-"*40)

                t0= datetime.now()

                # Gera alinhamentos
                aligns={}
                if cl_aln:= self.run_clustalw(f):
                    aligns["ClustalW"]= cl_aln
                if ms_aln:= self.run_muscle(f):
                    aligns["MUSCLE"]= ms_aln
                #alternado para funcionar
                msf_file= self.reference_dir/ f"{f.stem}.tfa"
                if msf_file.exists():
                    if bbb_aln:= self.convert_msf_to_aln(msf_file):
                        aligns["BAliBASE"]= bbb_aln
                else:
                    logger.warning(f"Reference MSF not found: {msf_file}")

                seq_res= {
                    "sequence_id": f.stem,
                    "methods": [],
                    "sp_raw": [],
                    "sp_norm": [],
                    "wsp_raw": [],
                    "wsp_norm": []
                }

                # Avalia
                for method, aln_file in aligns.items():
                    logger.info(f"\nEvaluating {method} alignment")
                    if scores:= self.evaluate_alignment(aln_file):
                        seq_res["methods"].append(method)
                        seq_res["sp_raw"].append(scores["sp_raw"])
                        seq_res["sp_norm"].append(scores["sp_norm"])
                        seq_res["wsp_raw"].append(scores["wsp_raw"])
                        seq_res["wsp_norm"].append(scores["wsp_norm"])

                        self._generate_alignment_statistics(aln_file, method)

                # Comparativos
                if len(seq_res["methods"])>1:
                    self._calculate_comparative_metrics(seq_res)

                results.append(seq_res)
                processed+=1
                dt= datetime.now()- t0
                logger.info(f"\nSequence completed in {dt}")

            except Exception as e:
                logger.error(f"Error processing {f}: {e}")
                errors+=1
                continue

        if results:
            self._save_results(results)
            self._generate_summary_report(results, processed, errors, total)
        else:
            logger.error("No results generated")

# -------------------------------------------
# MAIN
# -------------------------------------------
if __name__=="__main__":
    try:
        balibase_dir  = Path("/home/augusto/projects/multiOne/BAliBASE/RV30")
        reference_dir = Path("/home/augusto/projects/multiOne/BAliBASE/RV30")
        results_dir   = Path("/home/augusto/projects/multiOne/results")

        pipeline= SequenceAlignmentPipeline(
            balibase_dir= balibase_dir,
            reference_dir= reference_dir,
            results_dir= results_dir
        )
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
