# clustalw.py

import subprocess
import logging
from pathlib import Path

def run_clustalw(input_file: str, output_file: str, matrix_file: str) -> bool:
    """
    Executa ClustalW usando um arquivo de matriz específico.
    
    Args:
        input_file: Caminho para arquivo de sequências (FASTA)
        output_file: Caminho para salvar o alinhamento
        matrix_file: Caminho para arquivo da matriz de substituição
        
    Returns:
        bool: True se executou com sucesso, False caso contrário
    """
    try:
        # Encontra o executável do ClustalW
        clustalw_path = str(Path(__file__).parent / "../clustalw-2.1/src/clustalw2")
        
        # Prepara comando
        cmd = [
            clustalw_path,
            f"-INFILE={input_file}",
            f"-MATRIX={matrix_file}",
            "-ALIGN",
            "-OUTPUT=FASTA",
            f"-OUTFILE={output_file}",
            "-TYPE=PROTEIN"
        ]
        
        # Executa ClustalW
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Verifica se arquivo de saída foi gerado
        if not Path(output_file).exists():
            logging.error("ClustalW falhou em gerar arquivo de saída")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar ClustalW: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.error(f"ClustalW não encontrado em: {clustalw_path}")
        return False
    except Exception as e:
        logging.error(f"Erro inesperado no ClustalW: {str(e)}")
        return False