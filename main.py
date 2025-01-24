#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import os
from typing import List, Tuple
import numpy as np

# Add project root to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from memetic.matrix import AdaptiveMatrix
from memetic.memetic import MemeticAlgorithm

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptive_matrix.log'),
        logging.StreamHandler()
    ]
)

def run_clustalw(input_file: str, output_file: str, matrix_file: str) -> bool:
    """
    Executes ClustalW with a given matrix file. We use a temporary file for the matrix
    since ClustalW requires a file input, but our matrices are kept in memory.
    """
    clustalw_path = str(current_dir / "clustalw-2.1/src/clustalw2")
    
    cmd = [
        clustalw_path,
        f"-INFILE={input_file}",
        f"-MATRIX={matrix_file}",
        "-ALIGN",
        "-OUTPUT=FASTA",
        f"-OUTFILE={output_file}",
        "-TYPE=PROTEIN"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ClustalW error: {e.stderr}")
        return False

def get_bali_score(xml_file: str, alignment_file: str) -> float:
    """
    Calculates bali_score for an alignment. This function helps evaluate
    the quality of alignments produced by our adaptive matrices.
    """
    try:
        bali_score_path = str(current_dir / "baliscore" / "bali_score")
        result = subprocess.run(
            [bali_score_path, xml_file, alignment_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.split('\n'):
            if "CS score=" in line:
                return float(line.split('=')[1].strip())
        return 0.0
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Bali_score error: {e.stderr}")
        return 0.0

def evaluate_matrix(matrix: AdaptiveMatrix, 
                   xml_file: str, 
                   fasta_file: str,
                   num_evaluations: int = 1) -> float:
    """
    Evaluates a matrix multiple times to account for ClustalW variability.
    Uses temporary files for matrix and alignment storage, but the matrix
    itself remains primarily in memory.
    """
    scores = []
    # Create temporary directory if it doesn't exist
    temp_dir = current_dir / "memetic" / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    matrix_file = temp_dir / "temp_matrix.mat"
    matrix.to_clustalw_format(matrix_file)
    
    for i in range(num_evaluations):
        output_file = temp_dir / f"temp_aln_{i}.fasta"
        
        if run_clustalw(fasta_file, str(output_file), str(matrix_file)):
            score = get_bali_score(xml_file, str(output_file))
            if score > 0:
                scores.append(score)
                
        if output_file.exists():
            output_file.unlink()
            
    matrix_file.unlink()
    
    return np.mean(scores) if scores else 0.0

def save_best_matrix(matrix: AdaptiveMatrix, results_dir: Path) -> None:
    """
    Saves the best matrix found using the specified naming convention:
    'YYYY-MM-DD-HH-mm-AdaptivePAM.txt'
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"{timestamp}-AdaptivePAM.txt"
    output_path = results_dir / filename
    
    matrix.to_clustalw_format(output_path)
    logging.info(f"Best matrix saved to {output_path}")

def main():
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Setup directory paths relative to project root
    memetic_dir = project_root / "memetic"
    matrices_dir = memetic_dir / "matrices"
    results_dir = memetic_dir / "results"
    
    # Input files for BBA0142
    xml_file = str(project_root / "BAliBASE" / "RV100" / "BBA0142.xml")
    fasta_file = str(project_root / "BAliBASE" / "RV100" / "BBA0142.tfa")
    
    # Verify essential files exist
    essential_files = [
        (matrices_dir / "pam250.txt", "PAM250 matrix"),
        (Path(xml_file), "BBA0142 XML"),
        (Path(fasta_file), "BBA0142 sequences")
    ]
    
    for file_path, description in essential_files:
        if not file_path.exists():
            logging.error(f"Missing required file: {description} at {file_path}")
            sys.exit(1)
            
    logging.info("Starting memetic algorithm optimization")
    logging.info(f"Using PAM250 matrix from: {matrices_dir / 'pam250.txt'}")
    logging.info(f"Results will be saved to: {results_dir}")
    
    # Initialize and run memetic algorithm
    try:
        memetic = MemeticAlgorithm(
            population_size=50,
            elite_size=5,
            evaluation_function=lambda matrix: evaluate_matrix(
                matrix, 
                xml_file, 
                fasta_file, 
                num_evaluations=3
            ),
            xml_path=Path(xml_file)
        )
        
        best_matrix = memetic.run(
            generations=50,
            local_search_frequency=5,
            local_search_iterations=20,
            max_no_improve=10
        )
        
        # Save the best matrix found
        save_best_matrix(best_matrix, results_dir)
        
    except Exception as e:
        logging.error(f"Error during optimization: {str(e)}")
        sys.exit(1)
    
    logging.info("Optimization completed successfully")

if __name__ == "__main__":
    main()