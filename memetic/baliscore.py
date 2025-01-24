# baliscore.py

import subprocess
import logging
from pathlib import Path

def get_bali_score(xml_file: str, alignment_file: str) -> float:
    """
    Calcula o score de um alinhamento usando bali_score.
    
    Args:
        xml_file: Caminho para arquivo de referência XML
        alignment_file: Caminho para alinhamento a ser avaliado
        
    Returns:
        float: Score do alinhamento ou 0 se erro
    """
    try:
        # Encontra o executável do bali_score
        bali_score_path = str(Path(__file__).parent / "../baliscore/bali_score")
        
        # Executa bali_score
        result = subprocess.run(
            [bali_score_path, xml_file, alignment_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extrai score do output
        for line in result.stdout.split('\n'):
            if "CS score=" in line:
                score = float(line.split('=')[1].strip())
                logging.debug(f"BaliScore: {score}")
                return score
                
        logging.error("Score não encontrado na saída do bali_score")
        return 0.0
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar bali_score: {e.stderr}")
        return 0.0
    except FileNotFoundError:
        logging.error(f"bali_score não encontrado em: {bali_score_path}")
        return 0.0
    except Exception as e:
        logging.error(f"Erro inesperado no bali_score: {str(e)}")
        return 0.0