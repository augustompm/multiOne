#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def extract_reference_alignment(xml_path):
    """
    Extrai o alinhamento de referência do XML do BAliBASE Ref10,
    removendo as anotações e mantendo apenas o alinhamento.
    """
    try:
        xml_file = Path(xml_path)
        if not xml_file.exists():
            print(f"Erro: Arquivo {xml_path} não encontrado")
            return False
            
        output_file = xml_file.with_name(f"{xml_file.stem}_reference.fasta")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        sequences = []
        for seq in root.findall(".//sequence"):
            # Extrai nome da sequência
            seq_name = seq.find("seq-name")
            if seq_name is None or not seq_name.text:
                continue
            name = seq_name.text.strip()
            
            # Extrai dados da sequência
            seq_data = seq.find("seq-data")
            if seq_data is None or not seq_data.text:
                continue
                
            # Remove espaços em branco e quebras de linha
            seq_text = ''.join(seq_data.text.split())
            
            # Remove caracteres não-alinhamento (mantém apenas letras e gaps)
            seq_text = ''.join(c for c in seq_text if c.isalpha() or c == '-')
            
            # Cria registro FASTA
            record = SeqRecord(
                Seq(seq_text),
                id=name,
                name=name,
                description=""
            )
            sequences.append(record)
        
        if not sequences:
            print(f"Erro: Nenhuma sequência válida encontrada em {xml_path}")
            return False
            
        # Verifica se todos os alinhamentos têm o mesmo comprimento
        lengths = set(len(seq.seq) for seq in sequences)
        if len(lengths) > 1:
            print(f"Erro: Sequências com comprimentos diferentes encontradas")
            return False
            
        # Salva como FASTA
        SeqIO.write(sequences, output_file, "fasta")
        print(f"Arquivo criado com sucesso: {output_file}")
        print(f"Total de sequências: {len(sequences)}")
        print(f"Comprimento do alinhamento: {len(sequences[0].seq)}")
        return True
        
    except ET.ParseError as e:
        print(f"Erro ao parsear XML {xml_path}: {e}")
        return False
    except Exception as e:
        print(f"Erro ao processar {xml_path}: {str(e)}")
        import traceback
        print("Detalhes do erro:")
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 2:
        print("Uso: python extract_reference.py <arquivo.xml>")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    success = extract_reference_alignment(xml_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()