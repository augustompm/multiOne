#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import logging
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Caminho do executável ClustalW2
CLUSTALW_PATH = Path("/home/augusto/projects/multiOne/clustalw-2.1/src/clustalw2")

def run_clustalw_alignments(input_dir: Path, output_dir: Path):
    """
    Processa todos os arquivos .tfa em um diretório usando ClustalW2.
    """
    try:
        # Converte para Path se for string
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Verifica diretórios
        if not input_dir.is_dir():
            logging.error(f"Diretório de entrada não encontrado: {input_dir}")
            return False
            
        # Cria diretório de saída se não existir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verifica se ClustalW2 existe
        if not CLUSTALW_PATH.exists():
            logging.error(f"ClustalW2 não encontrado em: {CLUSTALW_PATH}")
            return False
            
        # Encontra todos os arquivos .tfa no diretório
        tfa_files = list(input_dir.glob("*.tfa"))
        
        if not tfa_files:
            logging.error(f"Nenhum arquivo .tfa encontrado em: {input_dir}")
            return False
            
        logging.info(f"Encontrados {len(tfa_files)} arquivos .tfa")
        
        # Processa cada arquivo
        success_count = 0
        error_count = 0
        start_time = datetime.now()
        
        for tfa_file in sorted(tfa_files):
            try:
                logging.info(f"\nProcessando: {tfa_file.name}")
                
                # Define arquivo de saída
                output_file = output_dir / f"{tfa_file.stem}_clustalw.fasta"
                
                # Comando ClustalW2
                cmd = [
                    str(CLUSTALW_PATH),
                    f"-INFILE={tfa_file}",
                    "-ALIGN",
                    "-OUTPUT=FASTA",
                    f"-OUTFILE={output_file}"
                ]
                
                # Executa ClustalW2
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Verifica se o arquivo foi gerado
                if output_file.exists() and output_file.stat().st_size > 0:
                    success_count += 1
                    logging.info(f"Sucesso: {output_file.name}")
                else:
                    error_count += 1
                    logging.error(f"Falha ao gerar alinhamento para: {tfa_file.name}")
                    
            except subprocess.CalledProcessError as e:
                error_count += 1
                logging.error(f"Erro ao processar {tfa_file.name}")
                logging.error(f"Saída de erro: {e.stderr}")
                continue
                
            except Exception as e:
                error_count += 1
                logging.error(f"Erro inesperado em {tfa_file.name}: {str(e)}")
                continue
        
        # Calcula tempo total
        elapsed_time = datetime.now() - start_time
        
        # Relatório final
        logging.info("\n" + "="*50)
        logging.info(f"Alinhamentos concluídos:")
        logging.info(f"Total de arquivos: {len(tfa_files)}")
        logging.info(f"Sucessos: {success_count}")
        logging.info(f"Erros: {error_count}")
        logging.info(f"Tempo total: {elapsed_time}")
        logging.info("="*50)
        
        return success_count > 0
        
    except Exception as e:
        logging.error(f"Erro ao processar diretório: {str(e)}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Uso: python3 run_clustalw.py <diretório_entrada> <diretório_saída>")
        print("Exemplo: python3 run_clustalw.py /home/augusto/projects/multiOne/BAliBASE/RV100 ./clustalw_results")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    success = run_clustalw_alignments(input_dir, output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()