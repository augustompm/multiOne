#!/usr/bin/env python3

from pathlib import Path
import logging
from memetic.matrix import AdaptiveMatrix

logging.basicConfig(level=logging.INFO)

def print_file_content(filepath: Path):
    """Imprime conteúdo do arquivo linha por linha com números de linha"""
    print(f"\nConteúdo de {filepath}:")
    print("-" * 80)
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            print(f"{i:3d}| {line.rstrip()}")
    print("-" * 80)

def main():
    # Cria diretório de teste se não existir
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Gera matriz usando nossa classe
    matrix = AdaptiveMatrix()
    our_matrix_file = test_dir / "our_matrix.mat"
    matrix.to_clustalw_format(our_matrix_file)
    
    # Caminho para uma matriz PAM250 que sabemos que funciona
    working_matrix = Path("memetic/matrices/pam250.txt")
    
    # Imprime ambos os formatos
    print("\n=== Formato que funciona ===")
    print_file_content(working_matrix)
    
    print("\n=== Nosso formato ===")
    print_file_content(our_matrix_file)
    
    # Análise básica do formato
    print("\nAnálise do formato:")
    
    with open(working_matrix) as f:
        working_lines = f.readlines()
    with open(our_matrix_file) as f:
        our_lines = f.readlines()
        
    # Compara número de linhas
    print(f"Número de linhas - Funcionando: {len(working_lines)}, Nossa: {len(our_lines)}")
    
    # Compara primeira linha (header)
    if len(working_lines) > 0 and len(our_lines) > 0:
        print("\nComparação do header:")
        print(f"Funcionando: '{working_lines[0].rstrip()}'")
        print(f"Nossa:       '{our_lines[0].rstrip()}'")
        
        # Conta espaços no header
        w_spaces = len([c for c in working_lines[0] if c == ' '])
        o_spaces = len([c for c in our_lines[0] if c == ' '])
        print(f"Espaços no header - Funcionando: {w_spaces}, Nossa: {o_spaces}")
    
    # Compara uma linha de valores
    if len(working_lines) > 1 and len(our_lines) > 1:
        print("\nComparação da primeira linha de valores:")
        print(f"Funcionando: '{working_lines[1].rstrip()}'")
        print(f"Nossa:       '{our_lines[1].rstrip()}'")
        
        # Conta números na linha
        w_nums = len([c for c in working_lines[1].split() if c.replace('-','').isdigit()])
        o_nums = len([c for c in our_lines[1].split() if c.replace('-','').isdigit()])
        print(f"Números na linha - Funcionando: {w_nums}, Nossa: {o_nums}")

if __name__ == "__main__":
    main()