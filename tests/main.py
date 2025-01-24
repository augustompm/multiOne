#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import os
from typing import List, Tuple, Dict
import json
import numpy as np

# Add project root to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from memetic.matrix import AdaptiveMatrix
from memetic.local_search import LocalSearch

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptive_matrix.log'),
        logging.StreamHandler()
    ]
)

class MetaheuristicConfig:
    """Manages configuration parameters for the metaheuristic algorithm."""
    
    def __init__(self):
        # Population parameters
        self.population_size = 30
        self.elite_size = 5
        self.generations = 100
        
        # Local search parameters
        self.local_search_frequency = 5  # Apply every N generations
        self.local_search_iterations = 20
        
        # VNS parameters
        self.max_no_improve = 10
        self.neighborhood_size = 3
        
        # Evaluation parameters
        self.evaluation_samples = 5  # Number of times to evaluate each matrix
        
    def save(self, filepath: Path):
        """Saves configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    @classmethod
    def load(cls, filepath: Path):
        """Loads configuration from JSON file."""
        config = cls()
        with open(filepath) as f:
            config.__dict__.update(json.load(f))
        return config

def run_clustalw(input_file: str, output_file: str, matrix_file: str) -> bool:
    """Executes ClustalW with a given matrix file."""
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
    """Calculates bali_score for an alignment."""
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
    Returns the mean score.
    """
    scores = []
    matrix_file = current_dir / "temp_matrix.mat"
    matrix.to_clustalw_format(matrix_file)
    
    for i in range(num_evaluations):
        output_file = current_dir / f"temp_aln_{i}.fasta"
        
        if run_clustalw(fasta_file, str(output_file), str(matrix_file)):
            score = get_bali_score(xml_file, str(output_file))
            if score > 0:
                scores.append(score)
                
        if output_file.exists():
            output_file.unlink()
            
    matrix_file.unlink()
    
    return np.mean(scores) if scores else 0.0

def main():
    # Load or create configuration
    config_file = current_dir / "config.json"
    if config_file.exists():
        config = MetaheuristicConfig.load(config_file)
    else:
        config = MetaheuristicConfig()
        config.save(config_file)
    
    # Setup paths for BBA0142
    xml_file = str(current_dir / "BAliBASE" / "RV100" / "BBA0142.xml")
    fasta_file = str(current_dir / "BAliBASE" / "RV100" / "BBA0142.tfa")
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Initialize population tracking
    best_scores = []
    mean_scores = []
    elite_matrices = []
    
    # Create initial population
    population = [AdaptiveMatrix() for _ in range(config.population_size)]
    
    # Initialize local search
    local_search = LocalSearch(population[0])  # Initialize with any matrix
    local_search.analyze_alignment(Path(xml_file))  # Analyze reference alignment
    
    # Main evolutionary loop
    for generation in range(config.generations):
        logging.info(f"\nGeneration {generation + 1}/{config.generations}")
        
        # Evaluate population
        scores = []
        for idx, matrix in enumerate(population):
            score = evaluate_matrix(
                matrix, 
                xml_file, 
                fasta_file, 
                config.evaluation_samples
            )
            scores.append(score)
            logging.info(f"Individual {idx + 1}: score = {score:.4f}")
            
        # Track statistics
        best_score = max(scores)
        mean_score = np.mean(scores)
        best_scores.append(best_score)
        mean_scores.append(mean_score)
        
        logging.info(f"Generation stats - Best: {best_score:.4f}, Mean: {mean_score:.4f}")
        
        # Select elite
        elite_indices = np.argsort(scores)[-config.elite_size:]
        elite_matrices = [population[i].copy() for i in elite_indices]
        
        # Apply local search to elite members
        if generation % config.local_search_frequency == 0:
            logging.info("Applying local search to elite members...")
            for matrix in elite_matrices:
                local_search.matrix = matrix
                local_search.vns_search(
                    lambda m: evaluate_matrix(m, xml_file, fasta_file),
                    max_iterations=config.local_search_iterations,
                    max_no_improve=config.max_no_improve
                )
        
        # Save best matrix
        if generation % 10 == 0:
            best_matrix = elite_matrices[-1]
            output_file = results_dir / f"best_matrix_gen_{generation}.mat"
            best_matrix.to_clustalw_format(output_file)
            
        # Create next generation (currently just copying elite)
        population = elite_matrices.copy()
        while len(population) < config.population_size:
            population.append(elite_matrices[0].copy())
            
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_matrix = results_dir / f"final_matrix_{timestamp}.mat"
    elite_matrices[-1].to_clustalw_format(final_matrix)
    
    # Save run statistics
    stats = {
        'best_scores': best_scores,
        'mean_scores': mean_scores,
        'config': config.__dict__
    }
    with open(results_dir / f"run_stats_{timestamp}.json", 'w') as f:
        json.dump(stats, f, indent=4)
        
    logging.info("\nOptimization completed")
    logging.info(f"Best score achieved: {best_scores[-1]:.4f}")

if __name__ == "__main__":
    main()