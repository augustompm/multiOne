#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import os
import glob

# Adiciona o diretório raiz ao path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from memetic.adaptive_matrix import AdaptiveMatrix

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_clustalw(input_file: str, output_file: str, matrix_file: str):
    """
    Executa o ClustalW usando a matriz PAM250 
    Args:
        input_file: Arquivo de entrada com as sequências
        output_file: Arquivo de saída para o alinhamento
        matrix_file: Arquivo com a matriz PAM250 no formato ClustalW
    """
    clustalw_path = str(root_dir / "clustalw-2.1/src/clustalw2")
    
    cmd = [
        clustalw_path,
        f"-INFILE={input_file}",
        f"-MATRIX={matrix_file}",
        "-ALIGN",
        "-OUTPUT=FASTA",
        f"-OUTFILE={output_file}",
        "-TYPE=PROTEIN"
    ]
    
    try:
        logging.info(f"Executando ClustalW com matriz PAM250...")
        logging.info(f"Arquivo de entrada: {input_file}")
        logging.info(f"Arquivo de saída: {output_file}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout:
            if "Alignment Score" in result.stdout:
                score = result.stdout.split("Alignment Score")[1].split("\n")[0].strip()
                logging.info(f"Alignment Score para {Path(input_file).stem}: {score}")
                
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ClustalW error para {Path(input_file).stem}: {e.stderr}")
        return False

def process_all_files(input_dir: str, output_dir: str):
    """
    Processa todos os arquivos .tfa do diretório de entrada
    """
    # Cria e salva a matriz PAM250 uma única vez
    matrix_file = str(Path(output_dir) / "pam250.mat")
    matrix = AdaptiveMatrix()
    matrix.to_clustalw_format(Path(matrix_file))
    logging.info(f"Matriz PAM250 salva em: {matrix_file}")
    
    # Lista todos os arquivos .tfa
    input_files = glob.glob(os.path.join(input_dir, "*.tfa"))
    total_files = len(input_files)
    
    logging.info(f"Encontrados {total_files} arquivos para processar")
    
    # Processa cada arquivo
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(input_files, 1):
        try:
            input_path = Path(input_file)
            output_file = str(Path(output_dir) / f"{input_path.stem}_clustalw_pam250.fasta")
            
            logging.info(f"Processando arquivo {i}/{total_files}: {input_path.stem}")
            
            if run_clustalw(input_file, output_file, matrix_file):
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logging.error(f"Erro processando {input_path.stem}: {str(e)}")
            failed += 1
            
    # Relatório final
    logging.info("\nRelatório Final:")
    logging.info(f"Total de arquivos processados: {total_files}")
    logging.info(f"Alinhamentos bem-sucedidos: {successful}")
    logging.info(f"Falhas: {failed}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 run_clustalw_pam250.py <input_dir> <output_dir>")
        print("Example: python3 run_clustalw_pam250.py /home/augusto/projects/multiOne/BAliBASE/RV100 ./clustalw_pam250_results")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        # Cria o diretório de saída se não existir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Processa todos os arquivos
        process_all_files(input_dir, output_dir)
        
    except Exception as e:
        logging.error(f"Erro fatal: {str(e)}")
        sys.exit(1)