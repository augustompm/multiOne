import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, NamedTuple
import numpy as np

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq

from .adaptive_matrix import AdaptiveMatrix

# Paths para programas externos
CLUSTALW_PATH = "/home/augusto/projects/multiOne/clustalw-2.1/src/clustalw2"
MUSCLE_PATH = "/home/augusto/projects/multiOne/muscle-5.3/src/muscle-linux"

class AlignmentScores(NamedTuple):
    """Estrutura para scores de alinhamento"""
    sp_raw: float      # SP Score bruto
    sp_norm: float     # SP Score normalizado [0..1]

class AlignmentEvaluator:
    """
    Avaliador que:
    1. Gera matriz em formato ClustalW/MUSCLE
    2. Alinha sequências usando matriz adaptativa
    3. Avalia usando SP e WSP scores
    """
    def __init__(self, matrix: AdaptiveMatrix):
        self.logger = logging.getLogger(__name__)
        self.matrix = matrix
        
        # Diretório de trabalho
        self.work_dir = Path("memetic/data")
        for d in ['clustalw', 'muscle', 'matrices']:
            (self.work_dir / d).mkdir(parents=True, exist_ok=True)

    def align_sequences(self, input_file: Path) -> Dict[str, Path]:
        """
        Alinha sequências usando a matriz adaptativa
        Args:
            input_file: Arquivo .tfa de entrada
        Returns:
            Dict com caminhos para arquivos .aln gerados
        """
        results = {}
        matrix_file = None
        
        try:
            # Salva matriz em formato ClustalW
            matrix_file = self.work_dir / "matrices" / "adaptive.mat"
            self.matrix.to_clustalw_format(matrix_file)
            
            # ClustalW
            clustalw_out = self.work_dir / "clustalw" / f"{input_file.stem}_clustalw.aln"
            cmd = [
                CLUSTALW_PATH,
                "-INFILE=" + str(input_file),
                "-MATRIX=" + str(matrix_file),
                "-OUTPUT=CLUSTAL",
                "-OUTFILE=" + str(clustalw_out),
                "-QUIET"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            results["clustalw"] = clustalw_out

            # MUSCLE
            muscle_out = self.work_dir / "muscle" / f"{input_file.stem}_muscle.aln"
            cmd = [
                MUSCLE_PATH,
                "-align", str(input_file),
                "-output", str(muscle_out)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Converte para clustal se necessário
            if muscle_out.exists():
                alignment = AlignIO.read(muscle_out, "fasta")
                AlignIO.write(alignment, str(muscle_out), "clustal")
            
            results["muscle"] = muscle_out
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro no alinhamento: {e}")
            if matrix_file and matrix_file.exists():
                matrix_file.unlink()
            return results

    def evaluate_alignment(self, alignment: MultipleSeqAlignment) -> AlignmentScores:
        """
        Avalia alinhamento usando SP score
        """
        try:
            # Calcula SP Score
            sp_raw = self._calculate_sp_score(alignment)
            sp_norm = self._normalize_sp_score(sp_raw, alignment)

            return AlignmentScores(
                sp_raw=sp_raw,
                sp_norm=sp_norm
            )

        except Exception as e:
            self.logger.error(f"Erro avaliando alinhamento: {e}")
            return AlignmentScores(0.0, 0.0)

    def _calculate_sp_score(self, alignment: MultipleSeqAlignment) -> float:
        """Calcula SP Score bruto"""
        try:
            score = 0.0
            n_seq = len(alignment)
            
            for i in range(n_seq-1):
                for j in range(i+1, n_seq):
                    seq1 = str(alignment[i].seq)
                    seq2 = str(alignment[j].seq)
                    
                    for col in range(len(seq1)):
                        # Pula se ambos são gaps terminais
                        if self._is_terminal_gap(seq1, col) or self._is_terminal_gap(seq2, col):
                            continue
                            
                        c1, c2 = seq1[col], seq2[col]
                        score += self.matrix.get_score(c1, c2)
                        
            return score
        except Exception as e:
            self.logger.error(f"Erro calculando SP score: {e}")
            return 0.0

    def _is_terminal_gap(self, sequence: str, pos: int) -> bool:
        """Verifica se posição é gap terminal"""
        if pos == 0 and sequence[0] == '-':
            return True
        if pos == len(sequence)-1 and sequence[-1] == '-':
            return True
        return False

    def _normalize_sp_score(self, score: float, alignment: MultipleSeqAlignment) -> float:
        """Normaliza SP Score para [0,1]"""
        try:
            if not alignment:
                return 0.0
                
            # Calcula máximo teórico
            max_score = 0.0
            n_seq = len(alignment)
            aln_len = alignment.get_alignment_length()
            
            # Máximo ocorre com matches perfeitos
            max_pair = max(self.matrix.score_cache.values())
            max_score = ((n_seq * (n_seq-1))/2) * aln_len * max_pair
                    
            if max_score == 0:
                return 0.0
                
            # Limita entre 0 e 1
            return max(0.0, min(1.0, score / max_score))
            
        except Exception as e:
            self.logger.error(f"Erro normalizando SP score: {e}")
            return 0.0