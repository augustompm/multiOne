import logging
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
import numpy as np
from Bio import AlignIO

from .adaptive_matrix import AdaptiveMatrix
from .evaluator import AlignmentEvaluator

class EvaluationResult(NamedTuple):
    """
    Estrutura para armazenar resultados da avaliação:
    
    matrix_id: Identificador único da matriz
    fo_original: FO original (SP score)
    fo_compare: FO comparativa (alpha*delta + beta*balibase)
    balibase_score: Score do alinhamento BAliBASE
    muscle_score: Score do alinhamento MUSCLE
    delta: Diferença BAliBASE - MUSCLE
    scores: Scores detalhados por método
    """
    matrix_id: str     
    fo_original: float 
    fo_compare: float  
    balibase_score: float  
    muscle_score: float    
    delta: float          
    scores: Dict          

class MemeticEngine:
    """
    Motor evolutivo que:
    1. Gerencia população de matrizes 
    2. Aplica busca local
    3. Avalia usando FO_compare e critérios biológicos
    4. Seleciona melhores soluções
    """
    def __init__(self,
                 xml_file: Path,
                 input_file: Path,
                 reference_file: Path,
                 alpha: float = 0.8,  # Peso da diferença BAliBASE-MUSCLE
                 beta: float = 0.2):  # Peso do score BAliBASE
                 
        self.logger = logging.getLogger(__name__)
        
        self.xml_file = xml_file
        self.input_file = input_file
        self.reference_file = reference_file
        self.alpha = alpha
        self.beta = beta
        
        # Valores mínimos aceitáveis
        self.min_delta = 0.0  # BAliBASE deve ser >= MUSCLE
        self.min_fo_compare = 0.03  # FO comparativa mínima
        
    def evaluate_matrix(self, matrix: AdaptiveMatrix, matrix_id: str) -> Optional[EvaluationResult]:
        """
        Avalia uma matriz considerando:
        1. FO original (SP)
        2. FO comparativa (diferença BAliBASE-MUSCLE)
        3. Scores individuais dos métodos
        """
        try:
            # Cria avaliador
            evaluator = AlignmentEvaluator(matrix)
            
            # Gera alinhamentos
            alignments = evaluator.align_sequences(self.input_file)
            
            # Avalia cada método
            method_scores = {}
            for method, aln_file in alignments.items():
                if aln_file.exists():
                    try:
                        alignment = AlignIO.read(aln_file, "clustal")
                        method_scores[method] = evaluator.evaluate_alignment(alignment)
                    except Exception as e:
                        self.logger.error(f"Erro avaliando {method}: {e}")
                        continue

            # Avalia BAliBASE    
            try:
                reference = AlignIO.read(self.reference_file, "clustal")
                method_scores['balibase'] = evaluator.evaluate_alignment(reference)
            except Exception as e:
                self.logger.error(f"Erro avaliando BAliBASE: {e}")
                return None

            # Calcula scores principais
            balibase_score = method_scores['balibase'].sp_norm  # Modificado: usar sp_norm ao invés de hybrid_norm
            muscle_score = method_scores['muscle'].sp_norm      # Modificado: usar sp_norm ao invés de hybrid_norm
            delta = balibase_score - muscle_score

            # Calcula FOs
            fo_original = method_scores['balibase'].sp_norm    # Modificado: usar apenas sp_norm
                         
            fo_compare = self.alpha * delta + self.beta * balibase_score

            return EvaluationResult(
                matrix_id=matrix_id,
                fo_original=fo_original,
                fo_compare=fo_compare,
                balibase_score=balibase_score,
                muscle_score=muscle_score,
                delta=delta,
                scores=method_scores
            )

        except Exception as e:
            self.logger.error(f"Erro avaliando matriz {matrix_id}: {e}")
            return None

    def evaluate_population(self, population: List[AdaptiveMatrix]) -> List[EvaluationResult]:
        """
        Avalia população completa e ordena por múltiplos critérios:
        1. Delta BAliBASE-MUSCLE >= 0
        2. FO_compare 
        3. FO_original como desempate
        """
        try:
            results = []
            
            # Avalia cada matriz
            for i, matrix in enumerate(population, 1):
                matrix_id = f"matrix_{i}"
                self.logger.info(f"\nAvaliando matriz {i}/{len(population)}")
                
                if result := self.evaluate_matrix(matrix, matrix_id):
                    results.append(result)
                    self.logger.info(
                        f"FO_compare: {result.fo_compare:.4f} "
                        f"Delta(BAli-MUS): {result.delta:.4f} "
                        f"FO_original: {result.fo_original:.4f}"
                    )

            # Filtra e ordena resultados
            sorted_results = self._sort_results(results)
            
            # Log das melhores soluções
            self.logger.info("\nMelhores soluções encontradas:")
            for i, result in enumerate(sorted_results[:5], 1):
                self.logger.info(
                    f"{i}. Matrix {result.matrix_id}: "
                    f"Delta={result.delta:.4f} "
                    f"FO_comp={result.fo_compare:.4f} "
                    f"FO_orig={result.fo_original:.4f}"
                )

            return sorted_results

        except Exception as e:
            self.logger.error(f"Erro avaliando população: {e}")
            return []

    def _sort_results(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        """
        Ordena resultados por múltiplos critérios:
        1. Prioriza BAliBASE >= MUSCLE (delta >= 0)
        2. Ordena por FO_compare
        3. Desempata por FO_original
        """
        try:
            # Separa em grupos por delta
            valid_results = []
            invalid_results = []
            
            for result in results:
                if result.delta >= self.min_delta and result.fo_compare >= self.min_fo_compare:
                    valid_results.append(result)
                else:
                    invalid_results.append(result)
                    
            # Ordena cada grupo
            valid_results.sort(
                key=lambda x: (x.fo_compare, x.fo_original),
                reverse=True
            )
            invalid_results.sort(
                key=lambda x: (x.fo_compare, x.fo_original),
                reverse=True
            )
            
            # Combina resultados priorizando válidos
            return valid_results + invalid_results
            
        except Exception as e:
            self.logger.error(f"Erro ordenando resultados: {e}")
            return results

    def get_best_matrix(self, results: List[EvaluationResult]) -> Optional[str]:
        """
        Retorna ID da melhor matriz que satisfaz:
        1. BAliBASE >= MUSCLE
        2. Maior FO_compare
        3. Maior FO_original em caso de empate
        """
        if not results:
            return None
            
        # Filtra por delta >= 0
        valid_results = [r for r in results 
                        if r.delta >= self.min_delta and 
                        r.fo_compare >= self.min_fo_compare]
                        
        if valid_results:
            return valid_results[0].matrix_id.split('_')[1]
            
        self.logger.warning("Nenhuma matriz encontrada com BAliBASE >= MUSCLE")
        return results[0].matrix_id.split('_')[1]

    def get_population_stats(self, results: List[EvaluationResult]) -> Dict:
        """Retorna estatísticas da população"""
        if not results:
            return {}
            
        deltas = [r.delta for r in results]
        fo_comps = [r.fo_compare for r in results]
        fo_origs = [r.fo_original for r in results]
        
        return {
            'n_total': len(results),
            'n_valid': len([r for r in results if r.delta >= self.min_delta]),
            'delta_mean': np.mean(deltas),
            'delta_std': np.std(deltas),
            'fo_compare_mean': np.mean(fo_comps),
            'fo_compare_std': np.std(fo_comps),
            'fo_original_mean': np.mean(fo_origs),
            'fo_original_std': np.std(fo_origs)
        }