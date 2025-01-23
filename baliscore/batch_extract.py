#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import logging

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_directory(dir_path: Path):
    """
    Processa todos os arquivos XML em um diretório e gera os alinhamentos de referência.
    """
    try:
        # Converte para Path se for string
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            logging.error(f"Diretório não encontrado: {dir_path}")
            return False
        
        # Encontra todos os arquivos XML no diretório
        xml_files = list(dir_path.glob("*.xml"))
        
        if not xml_files:
            logging.error(f"Nenhum arquivo XML encontrado em: {dir_path}")
            return False
            
        logging.info(f"Encontrados {len(xml_files)} arquivos XML")
        
        # Processa cada arquivo XML
        success_count = 0
        error_count = 0
        
        for xml_file in sorted(xml_files):
            try:
                logging.info(f"\nProcessando: {xml_file.name}")
                
                # Chama o extract_reference.py
                result = subprocess.run(
                    ["python3", "extract_reference.py", str(xml_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Verifica se o arquivo de referência foi criado
                ref_file = xml_file.with_name(f"{xml_file.stem}_reference.fasta")
                if ref_file.exists() and ref_file.stat().st_size > 0:
                    success_count += 1
                    logging.info(f"Sucesso: {ref_file.name}")
                else:
                    error_count += 1
                    logging.error(f"Falha ao gerar arquivo para: {xml_file.name}")
                    
            except subprocess.CalledProcessError as e:
                error_count += 1
                logging.error(f"Erro ao processar {xml_file.name}")
                logging.error(f"Saída de erro: {e.stderr}")
                continue
                
            except Exception as e:
                error_count += 1
                logging.error(f"Erro inesperado em {xml_file.name}: {str(e)}")
                continue
        
        # Relatório final
        logging.info("\n" + "="*50)
        logging.info(f"Processamento concluído:")
        logging.info(f"Total de arquivos: {len(xml_files)}")
        logging.info(f"Sucessos: {success_count}")
        logging.info(f"Erros: {error_count}")
        logging.info("="*50)
        
        return success_count > 0
        
    except Exception as e:
        logging.error(f"Erro ao processar diretório: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Uso: python3 batch_extract.py <diretório>")
        print("Exemplo: python3 batch_extract.py /path/to/RV100")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    success = process_directory(dir_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()