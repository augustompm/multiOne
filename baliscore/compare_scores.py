#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import logging
import csv
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_score(bali_output: str) -> float:
    """
    Extrai o CS score da saída do bali_score.
    """
    try:
        for line in bali_output.split('\n'):
            if "CS score=" in line:
                return float(line.split('=')[1].strip())
        return 0.0
    except Exception:
        return 0.0

def run_comparison(xml_dir: Path, clustalw_dir: Path, output_file: Path):
    """
    Compara alinhamentos de referência e ClustalW usando bali_score.
    """
    try:
        # Verifica diretórios
        if not xml_dir.is_dir():
            logging.error(f"Diretório XML não encontrado: {xml_dir}")
            return False
            
        if not clustalw_dir.is_dir():
            logging.error(f"Diretório ClustalW não encontrado: {clustalw_dir}")
            return False
            
        # Lista todos os arquivos XML
        xml_files = list(xml_dir.glob("*.xml"))
        
        if not xml_files:
            logging.error(f"Nenhum arquivo XML encontrado em: {xml_dir}")
            return False
            
        logging.info(f"Encontrados {len(xml_files)} arquivos XML")
        
        # Prepara arquivo CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['base_name', 'score_reference', 'score_clustalw'])
            
            # Processa cada arquivo
            for xml_file in sorted(xml_files):
                base_name = xml_file.stem
                logging.info(f"\nProcessando: {base_name}")
                
                # Define caminhos dos arquivos
                ref_file = xml_dir / f"{base_name}_reference.fasta"
                clustalw_file = clustalw_dir / f"{base_name}_clustalw.fasta"
                
                # Verifica se os arquivos existem
                if not ref_file.exists():
                    logging.error(f"Arquivo de referência não encontrado: {ref_file}")
                    continue
                    
                if not clustalw_file.exists():
                    logging.error(f"Arquivo ClustalW não encontrado: {clustalw_file}")
                    continue
                
                try:
                    # Executa bali_score para referência
                    ref_result = subprocess.run(
                        ["./bali_score", str(xml_file), str(ref_file)],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    ref_score = extract_score(ref_result.stdout)
                    
                    # Executa bali_score para ClustalW
                    clustalw_result = subprocess.run(
                        ["./bali_score", str(xml_file), str(clustalw_file)],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    clustalw_score = extract_score(clustalw_result.stdout)
                    
                    # Escreve resultados no CSV
                    writer.writerow([base_name, f"{ref_score:.3f}", f"{clustalw_score:.3f}"])
                    logging.info(f"Scores calculados - Ref: {ref_score:.3f}, ClustalW: {clustalw_score:.3f}")
                    
                except subprocess.CalledProcessError as e:
                    logging.error(f"Erro ao executar bali_score para {base_name}")
                    logging.error(f"Saída de erro: {e.stderr}")
                    writer.writerow([base_name, "ERROR", "ERROR"])
                    continue
                    
                except Exception as e:
                    logging.error(f"Erro inesperado processando {base_name}: {str(e)}")
                    writer.writerow([base_name, "ERROR", "ERROR"])
                    continue
        
        logging.info(f"\nResultados salvos em: {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Erro ao executar comparações: {str(e)}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Uso: python3 compare_scores.py <diretório_xml> <diretório_clustalw> <arquivo_saída.csv>")
        print("Exemplo: python3 compare_scores.py /home/augusto/projects/multiOne/BAliBASE/RV100 "
              "/home/augusto/projects/multiOne/baliscore/clustalw_results scores.csv")
        sys.exit(1)
    
    xml_dir = Path(sys.argv[1])
    clustalw_dir = Path(sys.argv[2])
    output_file = Path(sys.argv[3])
    
    success = run_comparison(xml_dir, clustalw_dir, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()