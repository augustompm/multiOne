#!/usr/bin/env python3

import subprocess
import logging
from pathlib import Path
import os
import sys

# Adiciona o diretório raiz ao PYTHONPATH
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from memetic.adaptive_matrix import AdaptiveMatrix

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_pam250_matrix(matrix_file: str):
    """
    Cria e salva a matriz PAM250 no formato do ClustalW
    """
    try:
        matrix = AdaptiveMatrix()  # Isso já inicializa com PAM250
        matrix.to_clustalw_format(Path(matrix_file))
        logging.info(f"Matriz PAM250 salva em: {matrix_file}")
        return True
    except Exception as e:
        logging.error(f"Erro criando matriz PAM250: {e}")
        return False

def run_clustalw_single(input_file: str, output_dir: str, matrix_file: str):
    """
    Executa ClustalW com PAM250 para um único arquivo
    """
    clustalw_path = str(root_dir / "clustalw-2.1/src/clustalw2")
    
    # Prepara os caminhos
    input_path = Path(input_file)
    output_file = str(Path(output_dir) / f"{input_path.stem}_clustalw_pam250.fasta")
    
    cmd = [
        clustalw_path,
        f"-INFILE={input_file}",
        f"-MATRIX={matrix_file}",
        "-ALIGN",
        "-OUTPUT=FASTA",
        f"-OUTFILE={output_file}",
        "-TYPE=PROTEIN"
    ]
    
    logging.info(f"Executando ClustalW para {input_path.stem}")
    logging.info(f"Comando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info("ClustalW executado com sucesso")
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro no ClustalW: {e.stderr}")
        return None

def run_bali_score(ref_file: str, test_file: str):
    """
    Executa bali_score e mostra resultado detalhado
    """
    bali_score_path = str(current_dir / "bali_score")
    cmd = [bali_score_path, ref_file, test_file]
    
    logging.info("\nExecutando bali_score")
    logging.info(f"Arquivo de referência: {ref_file}")
    logging.info(f"Arquivo de teste: {test_file}")
    logging.info(f"Comando: {' '.join(cmd)}")
    
    try:
        # Mostra conteúdo dos arquivos para debug
        logging.info("\nConteúdo do arquivo de referência:")
        with open(ref_file, 'r') as f:
            logging.info(f.read()[:500] + "...")
            
        logging.info("\nConteúdo do arquivo de teste:")
        with open(test_file, 'r') as f:
            logging.info(f.read()[:500] + "...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info("\nSaída do bali_score:")
        logging.info(result.stdout)
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro no bali_score: {e.stderr}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")

if __name__ == "__main__":
    # Configurações usando caminhos relativos ao diretório raiz
    test_file = str(root_dir / "BAliBASE/RV100/BBA0001.tfa")
    ref_file = str(root_dir / "BAliBASE/RV100/BBA0001_reference.fasta")
    output_dir = str(current_dir / "clustalw_pam250_results")
    matrix_file = str(current_dir / "pam250.mat")
    
    # Cria diretório de saída se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Cria a matriz PAM250
    if not create_pam250_matrix(matrix_file):
        logging.error("Falha ao criar matriz PAM250")
        sys.exit(1)
    
    # Executa ClustalW
    aligned_file = run_clustalw_single(test_file, output_dir, matrix_file)
    
    if aligned_file:
        # Executa bali_score
        run_bali_score(ref_file, aligned_file)