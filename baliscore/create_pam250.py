#!/usr/bin/env python3

from Bio.Align import substitution_matrices
from pathlib import Path

def create_pam250_matrix():
    """
    Cria a matriz PAM250 no formato do ClustalW:
    - Primeira linha: lista de aminoácidos
    - Linhas seguintes: aminoácido seguido dos scores
    """
    # Carrega a matriz do Biopython
    pam250 = substitution_matrices.load("PAM250")
    
    # Define a ordem dos aminoácidos (ordem padrão do ClustalW)
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    
    # Caminho para o arquivo
    matrix_path = Path("/home/augusto/projects/multiOne/clustalw-2.1/src/PAM250")
    
    with open(matrix_path, 'w') as f:
        # Primeira linha: aminoácidos separados por espaços
        f.write("   " + "  ".join(aa_order) + "\n")
        
        # Para cada aminoácido, escreve a linha com os scores
        for aa1 in aa_order:
            line = [aa1]  # Começa com o aminoácido
            for aa2 in aa_order:
                score = int(pam250[aa1, aa2])  # Converte para inteiro
                line.append(f"{score:3d}")  # Formata com 3 espaços
            f.write(" ".join(line) + "\n")

if __name__ == "__main__":
    create_pam250_matrix()