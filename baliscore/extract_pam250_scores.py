#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import logging
import csv
import glob
import os
import re
from Bio import AlignIO

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_to_clustal(input_file: str, output_file: str) -> bool:
    """
    Converte um arquivo de alinhamento para formato CLUSTAL
    """
    try:
        alignment = AlignIO.read(input_file, "fasta")
        AlignIO.write(alignment, output_file, "clustal")
        return True
    except Exception as e:
        logging.error(f"Erro convertendo {input_file} para CLUSTAL: {e}")
        return False

def calculate_bali_score(ref_file: str, test_file: str) -> tuple:
    """
    Calcula o bali_score entre o alinhamento de referência e o teste
    Retorna (sp_score, tc_score) ou (None, None) em caso de erro
    """
    try:
        # Cria arquivos temporários no formato CLUSTAL
        temp_ref = str(Path(ref_file).with_suffix('.aln'))
        temp_test = str(Path(test_file).with_suffix('.aln'))
        
        # Converte os arquivos
        if not convert_to_clustal(ref_file, temp_ref):
            return None, None
        if not convert_to_clustal(test_file, temp_test):
            return None, None
        
        # Executa bali_score com os arquivos convertidos
        cmd = ["./bali_score", temp_ref, temp_test]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Remove arquivos temporários
        os.remove(temp_ref)
        os.remove(temp_test)
        
        # Processa a saída para extrair os scores
        sp_score = None
        tc_score = None
        
        for line in result.stdout.split('\n'):
            if "SP=" in line:
                try:
                    sp_score = float(line.split('=')[1].strip())
                except:
                    sp_score = None
            elif "TC=" in line:
                try:
                    tc_score = float(line.split('=')[1].strip())
                except:
                    tc_score = None
                
        return sp_score, tc_score
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao calcular bali_score: {e.stderr}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        
    # Limpa arquivos temporários em caso de erro
    for temp_file in [temp_ref, temp_test]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
            
    return None, None

def compare_alignments(balibase_dir: str, pam250_dir: str, output_csv: str):
    """
    Compara os alinhamentos PAM250 com as referências e gera CSV
    """
    results = []
    
    # Lista todos os arquivos PAM250
    pam250_files = glob.glob(os.path.join(pam250_dir, "*_clustalw_pam250.fasta"))
    total_files = len(pam250_files)
    
    logging.info(f"Encontrados {total_files} arquivos para processar")
    
    for i, test_file in enumerate(sorted(pam250_files), 1):
        try:
            # Extrai o identificador BBAxxxx do nome do arquivo
            base_name = re.search(r'(BBA\d{4})', test_file).group(1)
            logging.info(f"Processando {i}/{total_files}: {base_name}")
            
            # Encontra o arquivo de referência correspondente
            ref_file = os.path.join(balibase_dir, f"{base_name}_reference.fasta")
            
            if not os.path.exists(ref_file):
                logging.warning(f"Arquivo de referência não encontrado: {ref_file}")
                continue
            
            # Calcula scores
            sp_score, tc_score = calculate_bali_score(ref_file, test_file)
            
            if sp_score is not None and tc_score is not None:
                results.append({
                    'test_case': base_name,
                    'sp_score': f"{sp_score:.3f}",
                    'tc_score': f"{tc_score:.3f}"
                })
                logging.info(f"{base_name}: SP={sp_score:.3f}, TC={tc_score:.3f}")
            
        except Exception as e:
            logging.error(f"Erro processando {base_name}: {e}")
            
    # Salva resultados em CSV
    try:
        if results:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['test_case', 'sp_score', 'tc_score'])
                writer.writeheader()
                writer.writerows(sorted(results, key=lambda x: x['test_case']))
                
            logging.info(f"\nResultados salvos em {output_csv}")
            logging.info(f"Total de casos processados com sucesso: {len(results)}")
            logging.info(f"Total de casos com falha: {total_files - len(results)}")
            
    except Exception as e:
        logging.error(f"Erro salvando CSV: {e}")

if __name__ == "__main__":
    # Diretórios e arquivo de saída
    balibase_dir = "/home/augusto/projects/multiOne/BAliBASE/RV100"
    pam250_dir = "./clustalw_pam250_results"
    output_csv = "pam250_scores.csv"
    
    try:
        compare_alignments(balibase_dir, pam250_dir, output_csv)
        
    except Exception as e:
        logging.error(f"Erro fatal: {str(e)}")
        sys.exit(1)