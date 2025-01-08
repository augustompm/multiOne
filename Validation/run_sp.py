#!/usr/bin/env python3
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import argparse
from pathlib import Path
import math

def calculate_sp_score(alignment: MultipleSeqAlignment) -> tuple[float, float]:
    """
    Calcula o SP score para um alinhamento múltiplo.
    
    Regras:
    - Resíduos iguais = 0
    - Resíduos diferentes ou gaps = 1
    
    Retorna:
    - sp_raw: Número total de diferenças encontradas
    - sp_norm: Score normalizado [0,1] (0 = melhor caso, 1 = pior caso)
    """
    if len(alignment) < 2:
        return 0.0, 0.0
        
    num_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()
    
    # Contador de diferenças totais
    total_differences = 0
    
    # Para cada coluna do alinhamento
    for col in range(aln_len):
        # Para cada par de sequências
        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                c1 = alignment[i, col]
                c2 = alignment[j, col]
                
                # Nova lógica para contar diferenças
                if c1 == c2 and c1 != '-':  # São iguais e não são gaps
                    continue  # Não conta diferença
                elif c1 != c2:  # São diferentes (incluindo quando um é gap)
                    total_differences += 1
    
    # Calcula valor máximo possível para normalização
    max_pairs = num_seqs * (num_seqs - 1) / 2
    max_score = max_pairs * aln_len
    
    # Normaliza o score para [0,1]
    normalized_score = total_differences / max_score if max_score > 0 else 1.0
    
    return total_differences, normalized_score

def main():
    parser = argparse.ArgumentParser(description='Calcula SP score para um alinhamento múltiplo')
    parser.add_argument('input_file', type=str, help='Arquivo de alinhamento em formato Clustal')
    args = parser.parse_args()
    
    try:
        # Lê o alinhamento
        alignment = AlignIO.read(args.input_file, "clustal")
        
        # Calcula os scores
        raw_score, norm_score = calculate_sp_score(alignment)
        
        print(f"\nResultados para {Path(args.input_file).name}:")
        print(f"SP Score (bruto): {raw_score}")
        print(f"SP Score (normalizado [0,1]): {norm_score:.4f}")
        print("Nota: 0 = melhor caso (tudo igual), 1 = pior caso (tudo diferente/gap)")
        
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    main()