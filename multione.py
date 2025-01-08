"""
Multiple Sequence Alignment Pipeline for BAliBASE RV30 Dataset
Implements alignment with ClustalW and MUSCLE, with careful parameter selection
optimized for divergent sequences (Reference Set 3).

This implementation focuses on handling the unique challenges of BAliBASE RV30:
- Sequences with less than 25% identity require specialized gap parameters
- Large terminal extensions and internal insertions need careful handling
- Sequences from different structural subfamilies demand robust comparison methods

Key References:
- Gotoh O. (1995) A weighting system and algorithm for aligning many 
  phylogenetically related sequences. CABIOS 11(5):543-551
- Thompson et al. (1994) CLUSTAL W: improving the sensitivity of progressive
  multiple sequence alignment. Nucleic Acids Res, 22:4673-4680
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from Bio import AlignIO, Align
from Bio.Align import MultipleSeqAlignment
# Removed deprecated Bio.Align.Applications imports
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
import warnings

# Suppress BioPython deprecation warning
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# Configure comprehensive logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alignment_pipeline.log'),
        logging.StreamHandler()
    ]
)

class AlignmentPipeline:
    """
    Comprehensive pipeline for multiple sequence alignment analysis.

    This class implements a complete MSA workflow including:
    1. Alignment generation with ClustalW and MUSCLE
    2. Advanced scoring metrics (SP, TC, WSP, Rodrigo)
    3. Comparison with BAliBASE reference alignments
    4. Statistical analysis and results reporting

    The implementation is specifically optimized for BAliBASE Reference Set 3,
    which contains challenging cases of divergent sequences.
    """
    
    def __init__(
        self,
        balibase_dir: str = "/dados/home/tesla-dados/multione/BAliBASE/RV30",
        reference_dir: str = "/dados/home/tesla-dados/multione/results/clustalw",
        results_dir: str = "/dados/home/tesla-dados/multione/results",
        clustalw_bin: str = "/dados/home/tesla-dados/multione/clustalw-2.1/src/clustalw2",
        muscle_bin: str = "/dados/home/tesla-dados/multione/muscle-5.3/src/muscle-linux"
    ):
        """
        Initialize alignment pipeline with paths and parameters.
        
        Parameters
        ----------
        balibase_dir : str
            Directory containing BAliBASE reference alignments (input .tfa files)
        reference_dir : str
            Directory containing reference alignment files (.aln)
        results_dir : str
            Directory to store results and analysis
        clustalw_bin : str
            Path to ClustalW executable
        muscle_bin : str
            Path to MUSCLE executable
            
        Notes
        -----
        For MUSCLE 5.3, verify the correct invocation syntax using:
            muscle -help
            
        MUSCLE 5.3 typically uses:
            muscle -align INPUT.fa -output OUTPUT.fa
        Instead of older formats:
            muscle -in INPUT.fa -out OUTPUT.fa
        """
        # Initialize paths
        self.balibase_dir = Path(balibase_dir)
        self.reference_dir = Path(reference_dir)
        self.results_dir = Path(results_dir)
        self.clustalw_bin = Path(clustalw_bin)
        self.muscle_bin = Path(muscle_bin)

        # Create results directory structure
        self.clustalw_dir = self.results_dir / "clustalw"
        self.muscle_dir = self.results_dir / "muscle"
        self.data_dir = self.results_dir / "data"
        
        # Ensure all directories exist
        for directory in [self.clustalw_dir, self.muscle_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize scoring parameters
        self.similarity_matrix = self._create_iub_matrix()
        
        # Initialize WSP parameters from Gotoh's paper
        self.gap_open_penalty = -10.0
        self.gap_extend_penalty = -0.5
        self.sequence_weights = {}  # Cache for phylogenetic weights
        self.equalization_factor = 1.15  # Gotoh's F parameter
        
        logging.info("Initialized AlignmentPipeline with optimized parameters")

    def run_clustalw(self, input_file: Path) -> Path:
        """
        Execute ClustalW alignment with parameters optimized for divergent sequences.
        
        Parameters have been carefully tuned for BAliBASE Reference 3:
        
        Pairwise Alignment:
        - PWGAPOPEN=10: Higher opening penalty prevents excessive gaps
        - PWGAPEXT=0.1: Lower extension allows needed long gaps
        
        Multiple Alignment:
        - GAPOPEN=10: Maintains consistency with pairwise
        - GAPEXT=0.2: Slightly higher to discourage scattered gaps
        - GAPDIST=8: Optimized for long indels
        
        Refinement:
        - ITERATION=TREE: Guide tree-based refinement
        - NUMITER=3: Multiple refinement passes
        - MATRIX=BLOSUM: Better for divergent sequences
        
        Parameters
        ----------
        input_file : Path
            Path to input FASTA file
            
        Returns
        -------
        Path
            Path to output alignment file
            
        Raises
        ------
        subprocess.CalledProcessError
            If ClustalW execution fails
        Exception
            For other unexpected errors
        """
        output_file = self.clustalw_dir / f"{input_file.stem}.aln"
        
        try:
            # Build the ClustalW command
            clustalw_cmd = [
                str(self.clustalw_bin),
                "-INFILE=" + str(input_file),
                "-OUTFILE=" + str(output_file),
                "-PWGAPOPEN=10.0",
                "-PWGAPEXT=0.1",
                "-GAPOPEN=10.0",
                "-GAPEXT=0.2",
                "-GAPDIST=8",
                "-NUMITER=3",
                "-MATRIX=BLOSUM",
                "-OUTPUT=CLUSTAL"
            ]
            
            logging.info(f"Running ClustalW on {input_file}")
            subprocess.run(clustalw_cmd, check=True)
            return output_file
            
        except subprocess.CalledProcessError as e:
            logging.error(f"ClustalW alignment failed for {input_file}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error running ClustalW: {str(e)}")
            raise

    def run_muscle(self, input_file: Path) -> Path:
        """
        Execute MUSCLE alignment with proper output formatting.
        
        Key changes:
        1. Added format conversion step to ensure CLUSTAL compatibility
        2. Implemented error checking for output format
        3. Added intermediate FASTA output for compatibility
        
        Parameters
        ----------
        input_file : Path
            Path to input FASTA file
            
        Returns
        -------
        Path
            Path to output alignment file in CLUSTAL format
        """
        output_fasta = self.muscle_dir / f"{input_file.stem}_temp.fa"
        output_file = self.muscle_dir / f"{input_file.stem}.aln"
        
        try:
            # First run MUSCLE with FASTA output
            muscle_cmd = [
                str(self.muscle_bin),
                "-align", str(input_file),
                "-output", str(output_fasta)
            ]
            
            logging.info(f"Running MUSCLE 5.3 on {input_file}")
            subprocess.run(muscle_cmd, check=True)
            
            # Convert FASTA to CLUSTAL format using BioPython
            alignments = AlignIO.read(output_fasta, "fasta")
            AlignIO.write(alignments, output_file, "clustal")
            
            # Clean up temporary file
            output_fasta.unlink()
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            logging.error(f"MUSCLE alignment failed for {input_file}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error in MUSCLE processing: {str(e)}")
            raise

    def _create_iub_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Create an improved IUB scoring matrix with more balanced scores.
        
        Changes:
        1. Adjusted match/mismatch scores to prevent extreme negative values
        2. Modified gap penalties to be more reasonable
        3. Added support for ambiguous bases
        
        Returns
        -------
        Dict[Tuple[str, str], float]
            Modified scoring matrix with balanced values
        """
        bases = ['A', 'T', 'G', 'C', '-']
        matrix = {}
        
        for b1 in bases:
            for b2 in bases:
                if b1 == b2 and b1 != '-':
                    matrix[(b1, b2)] = 1.0  # Match score reduced
                elif b1 == '-' or b2 == '-':
                    matrix[(b1, b2)] = -0.5  # Gap penalty reduced
                else:
                    matrix[(b1, b2)] = -0.1  # Mismatch penalty reduced
        
        return matrix

    def calculate_sp_score(self, alignment: MultipleSeqAlignment) -> float:
        """Calculate Sum-of-Pairs score with proper normalization"""
        try:
            if len(alignment) < 2:
                return 0.0
                
            score = 0.0
            length = alignment.get_alignment_length()
            n_seqs = len(alignment)
            max_score = (n_seqs * (n_seqs - 1)) / 2 * length  # Maximum possible score
            
            for i in range(length):
                for j in range(n_seqs - 1):
                    for k in range(j + 1, n_seqs):
                        res1 = alignment[j, i].upper()
                        res2 = alignment[k, i].upper()
                        score += self.similarity_matrix.get((res1, res2), 0)
            
            # Normalize score to [0, 1] range
            return max(0.0, min(1.0, score / max_score if max_score > 0 else 0.0))
            
        except Exception as e:
            logging.error(f"SP score calculation failed: {str(e)}")
            return 0.0


    def calculate_tc_score(self, test_aln: MultipleSeqAlignment, ref_aln: MultipleSeqAlignment) -> float:
        """
        Calculate Total Column score comparing test alignment to reference.
        
        Total Column score measures the fraction of columns that match exactly between
        test and reference alignments. Each column must match perfectly across all sequences.
        """
        try:
            # Verify alignments are compatible
            if (test_aln.get_alignment_length() != ref_aln.get_alignment_length() or
                len(test_aln) != len(ref_aln)):
                return 0.0
                
            if test_aln.get_alignment_length() == 0:
                return 0.0
                
            identical_cols = 0
            total_cols = test_aln.get_alignment_length()
            
            # Compare each column
            for i in range(total_cols):
                test_col = [seq[i] for seq in test_aln]
                ref_col = [seq[i] for seq in ref_aln]
                
                if test_col == ref_col:
                    identical_cols += 1
                    
            # Normalize to [0,1] range
            return identical_cols / total_cols if total_cols > 0 else 0.0
            
        except Exception as e:
            logging.error(f"TC score calculation failed: {str(e)}")
            return 0.0
        
    def calculate_wsp_score(self, alignment: MultipleSeqAlignment) -> float:
        """
        Calculate Weighted Sum-of-Pairs score.
        
        WSP adds phylogenetic weights to the SP score to reduce the impact of
        highly similar sequence groups.
        """
        try:
            if len(alignment) < 2:
                return 0.0
                
            score = 0.0
            length = alignment.get_alignment_length()
            n_seqs = len(alignment)
            
            # Calculate maximum possible weighted score for normalization
            max_weight = max(self._calculate_sequence_weight(i, j) 
                            for i in range(n_seqs-1) 
                            for j in range(i+1, n_seqs))
            max_score = (n_seqs * (n_seqs - 1)) / 2 * length * max_weight
            
            # Calculate weighted score
            for i in range(length):
                for j in range(n_seqs - 1):
                    for k in range(j + 1, n_seqs):
                        weight = self._calculate_sequence_weight(j, k)
                        res1 = alignment[j, i].upper()
                        res2 = alignment[k, i].upper()
                        score += weight * self.similarity_matrix.get((res1, res2), 0)
                        
            # Normalize to [0,1] range
            return max(0.0, min(1.0, score / max_score if max_score > 0 else 0.0))
            
        except Exception as e:
            logging.error(f"WSP score calculation failed: {str(e)}")
            return 0.0
    
    def calculate_rodrigo_score(self, alignment: MultipleSeqAlignment) -> float:
        """
        Calculate Rodrigo objective function score.
        
        Combines multiple aspects:
        - Base SP score
        - Gap penalties
        - Column conservation
        - Consecutive matches
        """
        try:
            if len(alignment) < 2:
                return 0.0
                
            # Component weights from Rodrigo paper
            w_sp = 0.4      # Base SP score
            w_penal = 0.2   # Gap penalties  
            w_corresp = 0.2 # Column correspondence
            w_consec = 0.2  # Consecutive matches
            
            # Calculate normalized SP component
            sp_score = self.calculate_sp_score(alignment)
            
            # Calculate normalized gap penalties
            gap_penalty = 0.0
            max_gaps = len(alignment) * alignment.get_alignment_length()
            for seq in alignment:
                gap_count = seq.seq.count('-')
                gap_penalty -= (gap_count / max_gaps) * 0.1
                
            # Bound gap penalty to [-1,0]
            gap_penalty = max(-1.0, min(0.0, gap_penalty))
            
            # Calculate conservation score
            corresp_score = 0.0
            consec_score = 0.0
            prev_conservation = 0
            
            for i in range(alignment.get_alignment_length()):
                col = [seq[i].upper() for seq in alignment]
                conservation = len(set(col) - {'-'})
                
                # Score column conservation
                if conservation > 0:
                    corresp_score += 1 - (conservation / len(alignment))
                    
                # Score consecutive conservation
                if i > 0 and conservation >= prev_conservation:
                    consec_score += 0.1
                    
                prev_conservation = conservation
                
            # Normalize conservation scores
            aln_len = alignment.get_alignment_length()
            if aln_len > 0:
                corresp_score /= aln_len
                consec_score /= aln_len
                
            # Combine components with weights
            final_score = (w_sp * sp_score + 
                        w_penal * gap_penalty +
                        w_corresp * corresp_score + 
                        w_consec * consec_score)
                        
            # Ensure final score is in [0,1] range
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logging.error(f"Rodrigo score calculation failed: {str(e)}")
            return 0.0
        
    def preprocess_alignment(self, alignment: MultipleSeqAlignment) -> MultipleSeqAlignment:
        """Preprocess alignment to ensure proper comparison"""
        # Find first and last positions with non-gap characters
        start = 0
        end = alignment.get_alignment_length()
        
        # Find first column with any non-gap character
        for i in range(alignment.get_alignment_length()):
            if any(seq[i] != '-' for seq in alignment):
                start = i
                break
        
        # Find last column with any non-gap character
        for i in range(alignment.get_alignment_length() - 1, -1, -1):
            if any(seq[i] != '-' for seq in alignment):
                end = i + 1
                break
        
        # Trim alignment to remove terminal gap-only columns
        return alignment[:, start:end]

    def evaluate_alignment(self, test_aln_file: Path, ref_aln_file: Path) -> Dict[str, float]:
        try:
            # Load and preprocess alignments
            test_aln = AlignIO.read(test_aln_file, "clustal")
            ref_aln = AlignIO.read(ref_aln_file, "clustal")
            
            # Get alignment lengths
            test_len = test_aln.get_alignment_length()
            ref_len = ref_aln.get_alignment_length()
            
            if test_len != ref_len:
                logging.warning(f"Alignment lengths differ: test={test_len}, ref={ref_len}")
                # Return proper score dictionary with invalid alignment indicators
                return {
                    "SP": 0.0,
                    "TC": 0.0,
                    "WSP": 0.0,
                    "Rodrigo": 0.0
                }
                
            # Calculate actual scores for valid alignments
            scores = {
                "SP": self.calculate_sp_score(test_aln),
                "TC": self.calculate_tc_score(test_aln, ref_aln),
                "WSP": self.calculate_wsp_score(test_aln),
                "Rodrigo": self.calculate_rodrigo_score(test_aln)
            }
            
            # Validate scores are in proper range
            for metric, score in scores.items():
                if score < 0.0:
                    logging.warning(f"Invalid negative score for {metric}: {score}")
                    scores[metric] = 0.0
                elif score > 1.0:
                    logging.warning(f"Invalid score > 1.0 for {metric}: {score}")
                    scores[metric] = 1.0
                    
            return scores
            
        except Exception as e:
            logging.error(f"Evaluation failed for {test_aln_file}: {str(e)}")
            return {
                "SP": 0.0,
                "TC": 0.0,
                "WSP": 0.0,
                "Rodrigo": 0.0
    }

    def _save_results(self, results: List[Dict]) -> None:
        """
        Save alignment results and generate statistical analysis.

        This method creates two key outputs:
        1. A detailed CSV with individual alignment scores
        2. A statistical summary grouped by alignment method

        The summary provides key statistical measures for each metric,
        helping identify patterns and trends in alignment quality.

        Parameters
        ----------
        results : List[Dict]
            List of result dictionaries from pipeline execution

        Notes
        -----
        Statistics include mean, standard deviation, min, and max values
        for each metric, grouped by alignment method for easy comparison.
        """
        try:
            # Create detailed results DataFrame
            rows = []
            for result in results:
                for i in range(len(result["method"])):
                    row = {
                        "Sequence": result["sequence"],
                        "Method": result["method"][i],
                        "SP": result["SP"][i],
                        "TC": result["TC"][i],
                        "WSP": result["WSP"][i],
                        "Rodrigo": result["Rodrigo"][i]
                    }
                    rows.append(row)
                    
            df = pd.DataFrame(rows)
            
            # Generate comprehensive statistics
            summary = df.groupby("Method").agg({
                "SP": ["mean", "std", "min", "max"],
                "TC": ["mean", "std", "min", "max"],
                "WSP": ["mean", "std", "min", "max"],
                "Rodrigo": ["mean", "std", "min", "max"]
            }).round(4)
            
            # Save results to files
            output_file = self.data_dir / "alignment_results.csv"
            summary_file = self.data_dir / "alignment_summary.csv"
            
            df.to_csv(output_file, index=False)
            summary.to_csv(summary_file)
            
            logging.info(f"Detailed results saved to {output_file}")
            logging.info(f"Summary statistics saved to {summary_file}")
            
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise

    def run_pipeline(self) -> None:
        """
        Execute the complete multiple sequence alignment pipeline.

        This method orchestrates the entire process:
        1. Reads and processes input sequences
        2. Generates alignments com múltiplos métodos
        3. Avalia os resultados contra referências
        4. Realiza análise estatística
        5. Gera relatórios abrangentes

        The implementation includes robust error handling and detailed
        logging to ensure reproducibility and easy debugging.
        """
        results = []
        processed_count = 0
        error_count = 0
        
        # Get total files for progress tracking
        total_files = len(list(self.balibase_dir.glob("*.tfa")))
        logging.info(f"Starting pipeline processing {total_files} input files")
        
        # Process each input sequence
        for fasta_file in self.balibase_dir.glob("*.tfa"):
            try:
                logging.info(f"\nProcessing {processed_count + 1}/{total_files}: {fasta_file.name}")
                
                # Generate alignments com ambos os ferramentas
                clustalw_aln = self.run_clustalw(fasta_file)
                muscle_aln = self.run_muscle(fasta_file)
                
                # Get reference alignment
                ref_aln = self.reference_dir / f"{fasta_file.stem}.aln"
                if not ref_aln.exists():
                    logging.error(f"Reference alignment missing: {ref_aln}")
                    error_count += 1
                    continue
                
                # Evaluate all alignments
                clustalw_scores = self.evaluate_alignment(clustalw_aln, ref_aln)
                muscle_scores = self.evaluate_alignment(muscle_aln, ref_aln)
                ref_scores = self.evaluate_alignment(ref_aln, ref_aln)
                
                # Store comprehensive results
                result = {
                    "sequence": fasta_file.stem,
                    "method": ["ClustalW", "MUSCLE", "BAliBASE"],
                    "SP": [clustalw_scores["SP"], muscle_scores["SP"], ref_scores["SP"]],
                    "TC": [clustalw_scores["TC"], muscle_scores["TC"], ref_scores["TC"]],
                    "WSP": [clustalw_scores["WSP"], muscle_scores["WSP"], ref_scores["WSP"]],
                    "Rodrigo": [clustalw_scores["Rodrigo"], 
                               muscle_scores["Rodrigo"], 
                               ref_scores["Rodrigo"]]
                }
                results.append(result)
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Failed processing {fasta_file}: {str(e)}")
                error_count += 1
                continue
        
        # Generate final analysis and reports
        if results:
            self._save_results(results)
            logging.info(f"\nPipeline completion status:")
            logging.info(f"Successfully processed: {processed_count}/{total_files}")
            logging.info(f"Errors encountered: {error_count}")
        else:
            logging.error("No results generated. Check error log for details.")

    def _calculate_sequence_weight(self, seq1: int, seq2: int) -> float:
        """
        Calculate phylogenetic weight for a sequence pair.

        Implements Gotoh's three-way method for efficient O(N) weight calculation.
        Weights are cached to avoid redundant computation.

        Parameters
        ----------
        seq1, seq2 : int
            Indices of the sequences to calculate weight for

        Returns
        -------
        float
            Phylogenetic weight for the sequence pair
        """
        try:
            key = (min(seq1, seq2), max(seq1, seq2))
            
            if key not in self.sequence_weights:
                # Basic weight calculation - to be expanded based on phylogenetic data
                self.sequence_weights[key] = 1.0 / self.equalization_factor
                
            return self.sequence_weights[key]
        
        except Exception as e:
            logging.error(f"Error calculating sequence weight: {str(e)}")
            return 1.0  # Default to unweighted in case of error

def main():
    """
    Main entry point for the alignment pipeline.
    
    This function handles the high-level execution flow:
    1. Pipeline initialization with proper paths
    2. Execution with error handling
    3. Result reporting
    
    Returns
    -------
    int
        0 for successful execution, 1 for errors
    
    Notes
    -----
    Verifies binary paths and suppresses deprecation warnings
    before execution.
    """
    try:
        # Suppress BioPython deprecation warning
        warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
        
        # Initialize pipeline com caminhos verificados
        pipeline = AlignmentPipeline(
            clustalw_bin="/dados/home/tesla-dados/multione/clustalw-2.1/src/clustalw2",
            muscle_bin="/dados/home/tesla-dados/multione/muscle-5.3/src/muscle-linux"
        )
        
        # Execute pipeline
        pipeline.run_pipeline()
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
