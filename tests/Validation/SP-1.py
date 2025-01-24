#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementação científica do SP (Sum-of-Pairs) score para avaliação de
alinhamentos múltiplos de proteínas, especialmente calibrado para o benchmark BAliBASE.

Agora com otimizações para reduzir chamadas repetitivas e acelerar o cálculo:
1) Armazenamos pam_min e pam_max no construtor
2) Usamos cache (score_cache) para (res_i, res_j)
3) Fazemos checagem prévia de colunas (poupando laços internos)
"""

import sys
import logging
from Bio import AlignIO
from Bio.Align import substitution_matrices
import numpy as np
from datetime import datetime

class SPScoreCalculator:
    def __init__(self):
        """
        Inicializa o calculador de SP score usando a matriz PAM250.
        
        - Carrega a matriz (Dayhoff et al., 1978)
        - Salva pam_min e pam_max (evitando chamá-los a cada par)
        - Cria cache de scores (evitando dicionário .get(...) em excesso)
        """
        self.matrix = substitution_matrices.load("PAM250")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Pré-calcula pontuação mínima e máxima para a matriz
        self.pam_min = min(self.matrix.values())
        self.pam_max = max(self.matrix.values())

        # Cache para (res_i, res_j) -> score
        self.score_cache = {}
        for (aa1, aa2), val in self.matrix.items():
            # Armazena no cache a pontuação do par (aa1, aa2)
            # e (aa2, aa1), pois a matriz é simétrica
            self.score_cache[(aa1, aa2)] = val
            self.score_cache[(aa2, aa1)] = val

    def compute_sequence_weights(self, alignment):
        """
        Implementa esquema de peso de sequência do CLUSTAL W (Thompson et al. 1994)
        usando distância evolutiva normalizada para [0,1].
        """
        num_seqs = len(alignment)
        weights = np.ones(num_seqs)
        
        for i in range(num_seqs):
            dists = []
            for j in range(num_seqs):
                if i != j:
                    dist = self._pam_distance(alignment[i], alignment[j])
                    dists.append(dist)
            # Peso baseado na divergência média
            weights[i] = np.mean(dists) if dists else 1.0

        # Normaliza soma total = 1
        total = np.sum(weights)
        if total > 0:
            weights /= total
        return weights

    def _pam_distance(self, seq1, seq2):
        """
        Distância normalizada [0..1] usando PAM250.
        Quanto maior a pontuação, menor a distância = 1 - normalizado.
        """
        aligned_positions = 0
        total_score = 0
        
        for pos in range(len(seq1)):
            if seq1[pos] != '-' and seq2[pos] != '-':
                # Recupera do cache
                score = self.score_cache.get((seq1[pos], seq2[pos]), 0)
                total_score += score
                aligned_positions += 1
                
        if aligned_positions == 0:
            return 1.0
        
        score_range = self.pam_max - self.pam_min
        if score_range == 0:
            return 0.0
        
        # Normaliza pontuação e inverte
        normalized_score = (total_score - self.pam_min * aligned_positions) / \
                           (score_range * aligned_positions)
        return 1.0 - normalized_score

    def compute_sp_score(self, alignment):
        """
        Calcula SP Score adaptado para [0..1], levando em conta:
        - Pesos de sequência (Thompson et al., 1994)
        - Gaps terminais não penalizados (Thompson et al., 1999)
        - Normalização por min/máx possíveis no alinhamento
        """
        weights = self.compute_sequence_weights(alignment)
        num_seqs = len(alignment)
        aln_length = alignment.get_alignment_length()

        # Soma bruta do SP
        total_sp_raw = 0.0
        total_pairs_count = 0

        # Soma mínima e máxima possíveis
        sp_min_possible = 0.0
        sp_max_possible = 0.0

        # Converte alignment para uma lista de strings
        # (Assim podemos acessar colunas mais rápido se quisermos)
        seq_records = [str(rec.seq) for rec in alignment]

        for pos in range(aln_length):
            # 1) Identificar quais sequências são válidas (sem gap terminal, sem gap nessa coluna)
            valid_idxs = []
            for i in range(num_seqs):
                if not self._is_terminal_gap(alignment[i], pos):
                    # Checar se este caractere não é '-'
                    if seq_records[i][pos] != '-':
                        valid_idxs.append(i)

            # Se menos de 2 sequências válidas, não há pares a pontuar
            if len(valid_idxs) < 2:
                continue

            column_real_score = 0.0
            column_min_score  = 0.0
            column_max_score  = 0.0
            column_pairs = 0

            # 2) Vamos iterar em pares (i, j) dentro da lista valid_idxs
            #    Evita laços duplos no array maior
            for idx_i in range(len(valid_idxs)-1):
                i = valid_idxs[idx_i]
                for idx_j in range(idx_i+1, len(valid_idxs)):
                    j = valid_idxs[idx_j]
                    # Recupera os resíduos
                    res_i = seq_records[i][pos]
                    res_j = seq_records[j][pos]

                    # Pega pontuação do par do cache
                    pair_score = self.score_cache.get((res_i, res_j), 0)

                    # Peso do par
                    pair_weight = weights[i] * weights[j]

                    # Soma real
                    column_real_score += pair_score * pair_weight
                    # Soma min
                    column_min_score  += self.pam_min * pair_weight
                    # Soma max
                    column_max_score  += self.pam_max * pair_weight

                    column_pairs += 1

            # Atualiza contadores globais
            if column_pairs > 0:
                total_sp_raw    += column_real_score
                sp_min_possible += column_min_score
                sp_max_possible += column_max_score
                total_pairs_count += column_pairs

        if total_pairs_count == 0:
            return 0.0

        # Normaliza no final
        denom = (sp_max_possible - sp_min_possible)
        if denom == 0:
            return 0.0

        sp_raw = total_sp_raw
        sp_norm = (sp_raw - sp_min_possible) / denom

        self.logger.info(f"SP Raw Score = {sp_raw:.3f}")
        self.logger.info(f"SP Min Possible = {sp_min_possible:.3f}")
        self.logger.info(f"SP Max Possible = {sp_max_possible:.3f}")
        self.logger.info(f"SP Normalized = {sp_norm:.4f}")

        return sp_norm

    def _is_terminal_gap(self, sequence, position):
        """
        Gaps terminais não penalizam (Thompson et al., 1999)
        """
        if position == 0:
            return sequence[position] == '-'
        elif position == len(sequence)-1:
            return sequence[position] == '-'
        return False

def main(aln_file):
    """
    Executa a leitura do alinhamento e computa o SP score normalizado [0..1].
    """
    calculator = SPScoreCalculator()
    
    try:
        alignment = AlignIO.read(aln_file, "clustal")
        sp_score_norm = calculator.compute_sp_score(alignment)
        
        print("\nAnálise BAliBASE")
        print("-" * 50)
        print(f"Arquivo: {aln_file}")
        print(f"Número de sequências: {len(alignment)}")
        print(f"Comprimento do alinhamento: {alignment.get_alignment_length()}")
        print(f"SP Score Normalizado: {sp_score_norm:.4f}")
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUso: python sp_score.py <arquivo.aln>")
        sys.exit(1)
    
    main(sys.argv[1])
