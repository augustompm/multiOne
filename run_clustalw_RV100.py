#!/usr/bin/env python3

import subprocess
from pathlib import Path
import logging
from datetime import datetime
import sys
from Bio import AlignIO
from Bio.Align import substitution_matrices
import os

# Instâncias do BAliBASE RV100 a serem processadas
INSTANCES = [
    'BBA0004', 'BBA0005', 'BBA0008', 'BBA0011', 'BBA0014', 
    'BBA0015', 'BBA0019', 'BBA0021', 'BBA0022', 'BBA0024',
    'BBA0080', 'BBA0126', 'BBA0133', 'BBA0142', 'BBA0148',
    'BBA0155', 'BBA0163', 'BBA0178', 'BBA0183', 'BBA0185',
    'BBA0192', 'BBA0201', 'BBA0218'
]

def setup_logging():
    """Configura logging para acompanhamento da execução"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = Path("biofit/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"clustalw_pam250_run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_pam250_matrix_file(output_dir: Path) -> Path:
    """
    Cria arquivo de matriz PAM250 no formato do ClustalW.
    
    Returns:
        Path do arquivo criado
    """
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_file = matrix_dir / "pam250.mat"
    
    # Carrega PAM250 do Biopython
    pam250 = substitution_matrices.load("PAM250")
    
    # Formato do ClustalW
    with open(matrix_file, 'w') as f:
        f.write("# PAM250 matrix in ClustalW format\n")
        f.write(f"# Number of entries = {len(pam250.alphabet)}\n")
        
        # Lista de aminoácidos
        f.write("   " + " ".join(pam250.alphabet) + "\n")
        
        # Matriz
        for i, aa1 in enumerate(pam250.alphabet):
            row = [aa1] + [str(int(pam250[i][j])) for j in range(len(pam250.alphabet))]
            f.write(" ".join(row) + "\n")
    
    return matrix_file

def run_clustalw(input_file: Path, output_dir: Path, matrix_file: Path) -> bool:
    """
    Executa ClustalW usando PAM250 para um arquivo de entrada
    
    Args:
        input_file: Arquivo .tfa de entrada
        output_dir: Diretório para salvar outputs
        matrix_file: Arquivo da matriz PAM250
        
    Returns:
        bool: True se execução foi bem sucedida
    """
    try:
        clustalw_path = Path("clustalw-2.1/src/clustalw2")
        
        if not clustalw_path.exists():
            raise FileNotFoundError(f"ClustalW não encontrado em: {clustalw_path}")
            
        base_name = output_dir / input_file.stem
        
        # Comando usando a matriz personalizada
        cmd_aln = [
            str(clustalw_path),
            "-INFILE=" + str(input_file),
            "-OUTFILE=" + str(base_name) + ".aln",
            "-OUTPUT=GDE",
            "-TYPE=PROTEIN",
            "-MATRIX=" + str(matrix_file),
            "-PWMATRIX=" + str(matrix_file)
        ]
        
        cmd_fasta = [
            str(clustalw_path),
            "-INFILE=" + str(input_file),
            "-OUTFILE=" + str(base_name) + ".fasta",
            "-OUTPUT=FASTA",
            "-TYPE=PROTEIN",
            "-MATRIX=" + str(matrix_file),
            "-PWMATRIX=" + str(matrix_file)
        ]
        
        # Define variável de ambiente para o ClustalW encontrar a matriz
        env = os.environ.copy()
        env["CLUSTALW_MATRIX"] = str(matrix_file.parent)
        
        # Executa comandos
        for cmd in [cmd_aln, cmd_fasta]:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            logging.debug(f"Saída do comando: {result.stdout}")
            
        if not (base_name.with_suffix('.aln').exists() and base_name.with_suffix('.fasta').exists()):
            raise FileNotFoundError("Arquivos de saída não foram gerados")
            
        # Verifica formato
        try:
            alignment = AlignIO.read(base_name.with_suffix('.fasta'), "fasta")
            logging.info(f"Alinhamento gerado com {len(alignment)} sequências, comprimento {alignment.get_alignment_length()}")
        except Exception as e:
            logging.error(f"Erro ao validar alinhamento: {e}")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar ClustalW para {input_file.name}: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Erro inesperado ao processar {input_file.name}: {str(e)}")
        return False

def main():
    setup_logging()
    logging.info("Iniciando execução do ClustalW (PAM250) para instâncias do RV100")
    
    balibase_dir = Path("BAliBASE/RV100")
    output_dir = Path("biofit/clustalw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cria arquivo da matriz PAM250
    try:
        matrix_file = create_pam250_matrix_file(output_dir)
        logging.info(f"Matriz PAM250 criada em: {matrix_file}")
    except Exception as e:
        logging.error(f"Erro ao criar matriz PAM250: {e}")
        sys.exit(1)
    
    successful = 0
    failed = 0
    
    for instance in INSTANCES:
        input_file = balibase_dir / f"{instance}.tfa"
        
        if not input_file.exists():
            logging.error(f"Arquivo não encontrado: {input_file}")
            failed += 1
            continue
            
        logging.info(f"Processando {instance} com PAM250")
        
        if run_clustalw(input_file, output_dir, matrix_file):
            successful += 1
            logging.info(f"ClustalW (PAM250) executado com sucesso para {instance}")
        else:
            failed += 1
            
    logging.info(f"\nProcessamento finalizado:")
    logging.info(f"Instâncias processadas com sucesso: {successful}")
    logging.info(f"Falhas: {failed}")

if __name__ == "__main__":
    main()