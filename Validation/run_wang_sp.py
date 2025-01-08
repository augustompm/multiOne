#!/usr/bin/env python3
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import argparse
from pathlib import Path

def print_column_comparison(col_index: int, seqs: list, pairs: list, differences: int):
    """Imprime os detalhes da comparação de uma coluna."""
    print(f"\nColuna {col_index + 1}:")
    print("Caracteres:", " ".join(seqs))
    print("Comparações de pares:")
    for (i, c1, j, c2, score) in pairs:
        print(f"  Seq {i+1} '{c1}' vs Seq {j+1} '{c2}': {score}")
    print(f"Total de diferenças na coluna: {differences}")

def calculate_detailed_sp_score(alignment: MultipleSeqAlignment) -> tuple[float, float]:
    """
    Calcula o SP score com detalhamento dos passos.
    
    Retorna:
    - sp_raw: Número total de diferenças encontradas
    - sp_norm: Score normalizado [0,1] (0 = melhor caso, 1 = pior caso)
    """
    if len(alignment) < 2:
        return 0.0, 0.0
        
    num_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()
    
    print(f"\nAnalisando alinhamento com {num_seqs} sequências de comprimento {aln_len}")
    
    # Calcula número máximo de comparações por coluna
    max_pairs = num_seqs * (num_seqs - 1) / 2
    print(f"Número máximo de pares por coluna: {max_pairs}")
    
    # Contador de diferenças totais
    total_differences = 0
    
    # Para cada coluna do alinhamento
    for col in range(aln_len):
        column_chars = []
        column_pairs = []
        column_differences = 0
        
        # Para cada par de sequências
        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                c1 = alignment[i, col]
                c2 = alignment[j, col]
                column_chars.extend([c1, c2])
                
                # Score 1 para diferentes ou gaps (minimização)
                score = 1 if (c1 != c2 or c1 == '-' or c2 == '-') else 0
                if score == 1:
                    column_differences += 1
                    
                column_pairs.append((i, c1, j, c2, score))
        
        # Imprime detalhes da coluna
        print_column_comparison(col, list(set(column_chars)), column_pairs, column_differences)
        total_differences += column_differences
    
    # Calcula valor máximo possível para normalização
    max_score = max_pairs * aln_len
    print(f"\nEstatísticas finais:")
    print(f"Total de diferenças encontradas: {total_differences}")
    print(f"Máximo possível de diferenças: {max_score}")
    
    # Normaliza o score para [0,1]
    normalized_score = total_differences / max_score if max_score > 0 else 1.0
    
    return total_differences, normalized_score

def main():
    parser = argparse.ArgumentParser(description='Calcula SP score detalhado para um alinhamento múltiplo')
    parser.add_argument('input_file', type=str, help='Arquivo de alinhamento em formato Clustal')
    args = parser.parse_args()
    
    try:
        # Lê o alinhamento
        alignment = AlignIO.read(args.input_file, "clustal")
        
        # Calcula os scores
        raw_score, norm_score = calculate_detailed_sp_score(alignment)
        
        print(f"\nResultados finais para {Path(args.input_file).name}:")
        print(f"SP Score (bruto): {raw_score}")
        print(f"SP Score (normalizado [0,1]): {norm_score:.4f}")
        print("Nota: 0 = melhor caso (tudo igual), 1 = pior caso (tudo diferente/gap)")
        
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    main()