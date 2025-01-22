#!/usr/bin/env python3
import logging
from pathlib import Path
from memetic.memetic_engine import MemeticEngine
from memetic.adaptive_matrix import AdaptiveMatrix
from memetic.evaluator import AlignmentEvaluator, AlignmentScores
from memetic.local_search import LocalSearch
from Bio import AlignIO

# =============================================
# PARÂMETROS AJUSTÁVEIS
# =============================================

# Paths para arquivos
BALIBASE_DIR = Path("BAliBASE/RV30")
INPUT_FILE = BALIBASE_DIR / "BB30002.tfa"
REFERENCE_FILE = BALIBASE_DIR / "BB30002.aln"
XML_FILE = BALIBASE_DIR / "BB30002.xml"
RESULTS_DIR = Path("results")

# Parâmetros da População
POPULATION_SIZE = 10
PERTURBATION_MIN = 0.05  # Mínima perturbação na PAM
PERTURBATION_MAX = 0.15  # Máxima perturbação na PAM
DIAGONAL_MIN = 1.1  # Mínimo reforço diagonal
DIAGONAL_MAX = 1.3  # Máximo reforço diagonal

# Probabilidades da Busca Local
P_NOTHING = 0.0      # Probabilidade de não alterar
P_PERTURB = 0.1      # Probabilidade de perturbação
P_CONSERVATION = 0.6  # Probabilidade busca conservação
P_STRUCTURAL = 0.3    # Probabilidade busca estrutural

# Pesos da Função Objetivo
ALPHA = 0.8  # Peso da diferença balibase-muscle
BETA = 0.2   # Peso do score balibase

# =============================================
# Configuração de Logging
# =============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_matrix(matrix: AdaptiveMatrix, input_file: Path, reference_file: Path) -> dict:
    """
    Avalia uma matriz:
    1. Alinha sequências usando a matriz
    2. Avalia alinhamentos gerados
    3. Compara com referência
    """
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
                    logger.info(f"SP: {scores.sp_norm:.4f}")
                except Exception as e:
                    logger.error(f"Erro avaliando {method}: {e}")
                    method_scores[method] = AlignmentScores(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Avalia alinhamento referência do BAliBASE
        try:
            reference = AlignIO.read(reference_file, "clustal")
            method_scores['balibase'] = evaluator.evaluate_alignment(reference)
            logger.info(f"\nScores para referência BAliBASE:")
            logger.info(f"SP: {method_scores['balibase'].sp_norm:.4f}")

        except Exception as e:
            logger.error(f"Erro avaliando referência: {e}")
            method_scores['balibase'] = AlignmentScores(0.0, 0.0, 0.0, 0.0, 0.0)
            
        return method_scores
            
    except Exception as e:
        logger.error(f"Erro avaliando matriz: {e}")
        return {}

def print_comparison(pam_scores: dict, adaptive_scores: dict):
    """Imprime comparação detalhada entre matrizes"""
    logger.info("\n" + "="*80)
    logger.info("MATRIX COMPARISON")
    logger.info("="*80)
    
    logger.info(f"{'Method/Matrix':<20} {'SP':<10}")
    logger.info("-"*50)  # Reduzido o tamanho da linha
    
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
        
        # Avalia população usando FO_compare
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
                f.write(f"  BAliBASE: {best_result.scores['balibase'].hybrid_norm:.4f}\n")
                f.write(f"  MUSCLE: {best_result.scores['muscle'].hybrid_norm:.4f}\n")
                f.write(f"  Delta: {best_result.scores['balibase'].hybrid_norm - best_result.scores['muscle'].hybrid_norm:.4f}\n")
                
                f.write("\nTODAS AS MATRIZES:\n")
                for i, result in enumerate(results, 1):
                    f.write(f"\nMatrix {i}:\n")
                    f.write(f"  FO_compare: {result.fo_compare:.4f}\n")
                    f.write(f"  FO original: {result.fo_original:.4f}\n")
                    f.write("  Scores:\n")
                    for method, scores in result.scores.items():
                        f.write(f"    {method}: {scores.hybrid_norm:.4f}\n")
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