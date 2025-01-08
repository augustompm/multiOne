import sys
from pathlib import Path
from Bio import AlignIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_clustal_file(file_path: Path) -> None:
    """
    Analisa um arquivo CLUSTAL para identificar possíveis problemas
    que podem causar falhas no parser do Biopython.
    """
    logger.info(f"\nDiagnosticando: {file_path}")
    
    # 1. Verificar se o arquivo existe e não está vazio
    if not file_path.exists():
        logger.error("Arquivo não encontrado!")
        return
    
    if file_path.stat().st_size == 0:
        logger.error("Arquivo está vazio!")
        return
        
    # 2. Examinar o cabeçalho e primeiras linhas
    try:
        with open(file_path, 'r') as f:
            # Ler as primeiras 10 linhas para análise
            lines = [line.rstrip() for line in f.readlines()[:10]]
            
        logger.info("=== Análise do Cabeçalho ===")
        if not lines:
            logger.error("Arquivo não contém linhas!")
            return
            
        # Verificar formato do cabeçalho CLUSTAL
        header = lines[0]
        logger.info(f"Cabeçalho encontrado: '{header}'")
        
        if not header.startswith(("CLUSTAL", "MUSCLE", "MSAPROBS")):
            logger.warning("Cabeçalho não começa com CLUSTAL/MUSCLE/MSAPROBS")
        
        # 3. Verificar estrutura básica do alinhamento
        logger.info("\n=== Estrutura do Alinhamento ===")
        seq_lines = [l for l in lines[1:] if l.strip() and not l.startswith(" ")]
        logger.info(f"Linhas de sequência encontradas: {len(seq_lines)}")
        
        for i, line in enumerate(seq_lines):
            logger.info(f"Linha {i+1}: {line[:50]}{'...' if len(line)>50 else ''}")
            
        # 4. Tentar parse com Biopython
        logger.info("\n=== Tentativa de Parse com Biopython ===")
        try:
            alignment = AlignIO.read(file_path, "clustal")
            logger.info(f"Parse bem sucedido! Encontradas {len(alignment)} sequências")
            logger.info(f"Comprimento do alinhamento: {alignment.get_alignment_length()}")
            
            # Mostrar primeiras sequências
            logger.info("\nPrimeiras sequências:")
            for record in alignment[:2]:
                logger.info(f"ID: {record.id}")
                logger.info(f"Seq: {record.seq[:50]}...")
                
        except Exception as e:
            logger.error(f"Erro no parse: {str(e)}")
            logger.error("Possíveis causas:")
            logger.error("- Cabeçalho incompatível")
            logger.error("- Formato de sequência incorreto")
            logger.error("- Caracteres inválidos")
            logger.error("- Inconsistência no comprimento das sequências")
            
        logger.info("\nDiagnóstico concluído!")
        
    except Exception as e:
        logger.error(f"Erro ao ler arquivo: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python diagnose_clustal.py <arquivo.aln>")
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    diagnose_clustal_file(file_path)