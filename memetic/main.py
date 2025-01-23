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
        Avalia alinhamento usando SP e WSP scores
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

    def _calculate_sequence_weights(self, alignment: MultipleSeqAlignment) -> np.ndarray:
        """
        Calcula pesos para cada sequência baseado em distâncias evolutivas
        """
        try:
            n_seq = len(alignment)
            weights = np.ones(n_seq)
            
            # Matriz de distâncias
            distances = np.zeros((n_seq, n_seq))
            for i in range(n_seq):
                for j in range(i+1, n_seq):
                    seq1 = str(alignment[i].seq)
                    seq2 = str(alignment[j].seq)
                    
                    # Distância como proporção de matches
                    matches = sum(1 for a, b in zip(seq1, seq2) 
                                if a != '-' and b != '-' and a == b)
                    total = sum(1 for a, b in zip(seq1, seq2) 
                              if a != '-' and b != '-')
                    
                    dist = 1.0 - (matches / total if total > 0 else 0)
                    distances[i,j] = dist
                    distances[j,i] = dist
                    
            # Peso como média das distâncias
            for i in range(n_seq):
                weights[i] = np.mean(distances[i,:])
                
            # Normaliza pesos
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            return weights
            
        except Exception as e:
            self.logger.error(f"Erro calculando pesos: {e}")
            return np.ones(len(alignment)) / len(alignment)

    def _calculate_wsp_score(self, alignment: MultipleSeqAlignment) -> float:
        """
        Calcula WSP Score considerando:
        1. Pesos de sequência baseados em distância evolutiva
        2. Score de substituição da matriz adaptativa
        3. Penalização de gaps
        """
        try:
            # Calcula pesos de sequência
            weights = self._calculate_sequence_weights(alignment)
            
            n_seq = len(alignment)
            aln_len = alignment.get_alignment_length()
            wsp_score = 0.0
            
            # Para cada coluna
            for col in range(aln_len):
                col_score = 0.0
                
                # Para cada par de sequências
                for i in range(n_seq-1):
                    seq1 = str(alignment[i].seq)
                    for j in range(i+1, n_seq):
                        seq2 = str(alignment[j].seq)
                        
                        # Pula se é gap terminal
                        if self._is_terminal_gap(seq1, col) or self._is_terminal_gap(seq2, col):
                            continue
                        
                        # Score do par na coluna
                        res1 = seq1[col]
                        res2 = seq2[col]
                        
                        # Pula se ambos são gaps
                        if res1 == '-' and res2 == '-':
                            continue
                            
                        # Score ponderado
                        pair_score = self.matrix.get_score(res1, res2)
                        weighted_score = pair_score * weights[i] * weights[j]
                        
                        col_score += weighted_score
                        
                wsp_score += col_score
                
            return wsp_score
            
        except Exception as e:
            self.logger.error(f"Erro calculando WSP score: {e}")
            return 0.0
            
    def _normalize_wsp_score(self, score: float, alignment: MultipleSeqAlignment) -> float:
        """
        Normaliza WSP Score para [0,1] usando:
        - Máximo teórico (todos match perfeitos)
        - Mínimo teórico (todos mismatch)
        """
        try:
            if not alignment:
                return 0.0
                
            weights = self._calculate_sequence_weights(alignment)
            n_seq = len(alignment)
            aln_len = alignment.get_alignment_length()
            
            # Máximo teórico: todos matches perfeitos
            max_score = 0.0
            best_match = max(self.matrix.score_cache.values())
            
            # Mínimo teórico: todos mismatches
            min_score = 0.0
            worst_mismatch = min(self.matrix.score_cache.values())
            
            # Soma ponderada para todas as colunas
            for i in range(n_seq-1):
                for j in range(i+1, n_seq):
                    weight_prod = weights[i] * weights[j]
                    max_score += aln_len * best_match * weight_prod
                    min_score += aln_len * worst_mismatch * weight_prod
            
            # Evita divisão por zero
            if max_score == min_score:
                return 0.0
                
            # Normaliza entre min e max
            norm_score = (score - min_score) / (max_score - min_score)
            
            # Limita entre 0 e 1
            return max(0.0, min(1.0, norm_score))
            
        except Exception as e:
            self.logger.error(f"Erro normalizando WSP score: {e}")
            return 0.0

def evaluate_matrix(matrix: AdaptiveMatrix, input_file: Path, reference_file: Path) -> dict:
    try:
        logger.info(f"\nAvaliando matriz...")
        
        # Cria avaliador com a matriz
        evaluator = AlignmentEvaluator(matrix)
        
        # Gera alinhamentos usando a matriz
        alignments = evaluator.align_sequences(input_file)
        
        # Dicionário para armazenar scores de cada método
        method_scores = {}
        
        # Avalia ClustalW e MUSCLE
        for method, aln_file in alignments.items():
            if aln_file.exists():
                try:
                    alignment = AlignIO.read(aln_file, "clustal")
                    scores = evaluator.evaluate_alignment(alignment)
                    method_scores[method] = scores
                    logger.info(f"\nScores para {method}:")
                    logger.info(f"SP: {scores.sp_norm:.4f}")  # Remover linha do WSP
                except Exception as e:
                    logger.error(f"Erro avaliando {method}: {e}")
                    method_scores[method] = AlignmentScores(0.0, 0.0)  # Atualizar construtor
        
        # Avalia alinhamento referência do BAliBASE
        try:
            reference = AlignIO.read(reference_file, "clustal")
            method_scores['balibase'] = evaluator.evaluate_alignment(reference)
            logger.info(f"\nScores para referência BAliBASE:")
            logger.info(f"SP: {method_scores['balibase'].sp_norm:.4f}")  # Remover linha do WSP
        except Exception as e:
            logger.error(f"Erro avaliando referência: {e}")
            method_scores['balibase'] = AlignmentScores(0.0, 0.0)  # Atualizar construtor
            
        return method_scores
            
    except Exception as e:
        logger.error(f"Erro avaliando matriz: {e}")
        return {}

def print_comparison(pam_scores: dict, adaptive_scores: dict):
    """Imprime comparação detalhada entre matrizes"""
    logger.info("\n" + "="*80)
    logger.info("MATRIX COMPARISON")
    logger.info("="*80)
    
    # Header
    logger.info(f"{'Method/Matrix':<20} {'SP':<10}")  # Remover WSP e Hybrid
    logger.info("-"*50)  # Reduzir tamanho da linha
    
    # Scores para cada método com PAM250
    for method in ['clustalw', 'muscle', 'balibase']:
        if method in pam_scores:
            logger.info(f"{f'PAM250 {method}':<20} "
                       f"{pam_scores[method].sp_norm:>10.4f}")
    
    logger.info("-"*50)
    
    # Scores para cada método com matriz adaptativa
    for method in ['clustalw', 'muscle', 'balibase']:
        if method in adaptive_scores:
            logger.info(f"{f'AdaptivePAM {method}':<20} "
                       f"{adaptive_scores[method].sp_norm:>10.4f}")
    
    logger.info("-"*50)
    logger.info("% Improvement over PAM250:")
    
    # Calcula melhoria percentual para cada método
    for method in ['clustalw', 'muscle', 'balibase']:
        if method in pam_scores and method in adaptive_scores:
            logger.info(f"\n{method.upper()}:")
            pam = pam_scores[method]
            adap = adaptive_scores[method]
            
            # SP Score
            if pam.sp_norm == 0:
                sp_text = "undefined" if adap.sp_norm != 0 else "no change"
            else:
                sp_diff = ((adap.sp_norm - pam.sp_norm) / pam.sp_norm) * 100
                sp_text = f"{sp_diff:>.2f}%"
            
            logger.info(f"{'SP:':<12} {sp_text:>10}")
    
    logger.info("-"*50)

def main():
    try:
        logger.info("\nMA-BioFit Iniciando ...")

        # Inicializa busca local com parâmetros do topo
        local_search = LocalSearch(
            xml_file=XML_FILE,
            p_nothing=P_NOTHING,
            p_perturb=P_PERTURB,
            p_conservation=P_CONSERVATION,
            p_structural=P_STRUCTURAL
        )
        
        # Inicializa motor memético
        memetic = MemeticEngine(
            xml_file=XML_FILE,
            input_file=INPUT_FILE,
            reference_file=REFERENCE_FILE,
            alpha=ALPHA,
            beta=BETA
        )
        
        # Cria PAM250 original e avalia
        logger.info("\nCriando e avaliando PAM250 original...")
        pam250 = AdaptiveMatrix()
        pam_scores = evaluate_matrix(pam250, INPUT_FILE, REFERENCE_FILE)
        
        # Gera população inicial
        logger.info("\nGerando população inicial...")
        population = pam250.generate_population(
            size=POPULATION_SIZE,
            perturbation_range=(PERTURBATION_MIN, PERTURBATION_MAX),
            diagonal_range=(DIAGONAL_MIN, DIAGONAL_MAX)
        )
        
        # Avalia população usando FO_compare (baseado em SP)
        logger.info("\nAvaliando população inicial...")
        results = memetic.evaluate_population(population)
        
        # Identifica melhor matriz
        if best_idx := memetic.get_best_matrix(results):
            best_matrix = population[int(best_idx)-1]
            best_result = results[0]
            
            # Compara com PAM250 original
            logger.info("\nComparando melhor matriz com PAM250...")
            print_comparison(pam_scores, best_result.scores)
            
            # Salva melhor matriz e resultados
            RESULTS_DIR.mkdir(exist_ok=True)
            
            best_matrix.save(RESULTS_DIR / "best_matrix.npy")
            logger.info(f"\nMelhor matriz salva em {RESULTS_DIR}/best_matrix.npy")
            
            # Salva análise completa
            with open(RESULTS_DIR / "analysis.txt", "w") as f:
                f.write("ANÁLISE DETALHADA\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("MELHOR MATRIZ:\n")
                f.write(f"FO_compare: {best_result.fo_compare:.4f}\n")
                f.write(f"FO original: {best_result.fo_original:.4f}\n")
                f.write("\nDiferença BAliBASE vs MUSCLE:\n")
                f.write(f"  BAliBASE: {best_result.scores['balibase'].sp_norm:.4f}\n")
                f.write(f"  MUSCLE: {best_result.scores['muscle'].sp_norm:.4f}\n")
                f.write(f"  Delta: {best_result.scores['balibase'].sp_norm - best_result.scores['muscle'].sp_norm:.4f}\n")
                
                f.write("\nTODAS AS MATRIZES:\n")
                for i, result in enumerate(results, 1):
                    f.write(f"\nMatrix {i}:\n")
                    f.write(f"  FO_compare: {result.fo_compare:.4f}\n")
                    f.write(f"  FO original: {result.fo_original:.4f}\n")
                    f.write("  Scores:\n")
                    for method, scores in result.scores.items():
                        f.write(f"    {method}: {scores.sp_norm:.4f}\n")
                    f.write("-" * 40 + "\n")
            
            logger.info("\nAvaliação completa!")
            logger.info(f"Análise detalhada salva em {RESULTS_DIR}/analysis.txt")
            
        else:
            logger.error("Nenhuma matriz válida encontrada!")
            
    except Exception as e:
        logger.error(f"Avaliação falhou: {e}")
        raise

if __name__ == "__main__":
    main()