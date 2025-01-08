#!/usr/bin/env python3
from Bio import AlignIO
import sys

def convert_fasta_to_clustal(input_file: str, output_file: str):
    """Converte um arquivo de alinhamento do formato FASTA para CLUSTAL."""
    try:
        # Lê o alinhamento em formato FASTA
        alignment = AlignIO.read(input_file, "fasta")
        
        # Escreve o alinhamento em formato CLUSTAL
        AlignIO.write(alignment, output_file, "clustal")
        print(f"Arquivo convertido com sucesso: {output_file}")
        
    except Exception as e:
        print(f"Erro ao converter arquivo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python3 convert_aln.py arquivo_entrada.fa arquivo_saida.aln")
        sys.exit(1)
        
    convert_fasta_to_clustal(sys.argv[1], sys.argv[2])