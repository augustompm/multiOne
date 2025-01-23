#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def extract_reference_alignment(xml_path):
    """
    Extrai o alinhamento de referência de um arquivo XML do BAliBASE 4.0
    e salva como FASTA.
    
    Args:
        xml_path: Caminho para o arquivo XML
    """
    try:
        # Verifica se o arquivo existe
        xml_file = Path(xml_path)
        if not xml_file.exists():
            print(f"Erro: Arquivo {xml_path} não encontrado")
            return False
            
        # Nome do arquivo de saída
        output_file = xml_file.with_name(f"{xml_file.stem}_reference.fasta")
        
        # Parse o XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Encontra o elemento de alinhamento
        alignment = root.find(".//alignment")
        if alignment is None:
            print(f"Erro: Elemento 'alignment' não encontrado em {xml_path}")
            return False
            
        # Extrai as sequências
        sequences = []
        for seq in alignment.findall(".//sequence"):
            # Pega o nome e a sequência
            name = seq.get('name', '')
            seq_text = seq.find('seq').text.strip()
            
            # Cria um registro FASTA
            record = SeqRecord(
                Seq(seq_text),
                id=name,
                name=name,
                description=""
            )
            sequences.append(record)
        
        # Verifica se encontrou sequências
        if not sequences:
            print(f"Erro: Nenhuma sequência encontrada em {xml_path}")
            return False
            
        # Salva como FASTA
        SeqIO.write(sequences, output_file, "fasta")
        print(f"Arquivo criado com sucesso: {output_file}")
        return True
        
    except ET.ParseError as e:
        print(f"Erro ao parsear XML {xml_path}: {e}")
        return False
    except Exception as e:
        print(f"Erro ao processar {xml_path}: {e}")
        return False

def main():
    # Verifica argumentos
    if len(sys.argv) != 2:
        print("Uso: python extract_reference.py <arquivo.xml>")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    success = extract_reference_alignment(xml_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()