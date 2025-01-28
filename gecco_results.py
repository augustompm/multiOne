#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import subprocess
import re
import logging
import csv
from typing import Dict, List, Tuple, Optional
from datetime import datetime 
from collections import defaultdict

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/home/augusto/projects/multiOne')
CLUSTALW_PATH = PROJECT_ROOT / 'clustalw-2.1/src/clustalw2'
BALISCORE_PATH = PROJECT_ROOT / 'baliscore/bali_score'
BALIBASE_DIR = PROJECT_ROOT / 'BAliBASE/RV100'
MATRIX_DIR = PROJECT_ROOT / 'multi_memetic/results'
TEMP_DIR = PROJECT_ROOT / 'multi_memetic/temp'
EVAL_DIR = PROJECT_ROOT / 'multi_memetic/evaluations'

# Criar diretórios necessários
for dir_path in [TEMP_DIR, EVAL_DIR]:
   dir_path.mkdir(exist_ok=True, parents=True)

# Limpar diretório temporário
for f in TEMP_DIR.glob('*'):
   f.unlink()

INSTANCES = [
   'BBA0005', 'BBA0014', 'BBA0019', 'BBA0022', 
   'BBA0080', 'BBA0126', 'BBA0142', 'BBA0155',
   'BBA0183', 'BBA0185', 'BBA0192', 'BBA0201',
   'BBA0133', 'BBA0004', 'BBA0178', 'BBA0024', 
   'BBA0008', 'BBA0015', 'BBA0011', 'BBA0163', 
   'BBA0148', 'BBA0021', 'BBA0218'
]

def get_latest_matrices() -> Dict[str, Path]:
   """Encontra as matrizes mais recentes entre results e evaluation runs"""
   matrix_files = {
       'HIGH': 'BioFit_HIGH_*.mat',
       'MEDIUM': 'BioFit_MEDIUM_*.mat',
       'LOW': 'BioFit_LOW_*.mat',
       'combined': 'BioFit_combined_*.mat'
   }
   
   # Procura primeiro no diretório results
   matrix_paths = {}
   for matrix_type, pattern in matrix_files.items():
       files = list(MATRIX_DIR.glob(pattern))
       if files:
           latest = max(files, key=lambda x: x.stat().st_mtime)
           matrix_paths[matrix_type] = latest
           
   # Se não encontrou todas, procura nos runs anteriores
   if len(matrix_paths) < 4:
       # Lista todos os diretórios run_* em ordem decrescente
       run_dirs = sorted([d for d in EVAL_DIR.glob('run_*') if d.is_dir()], 
                       key=lambda x: int(x.name.split('_')[1]), 
                       reverse=True)
       
       # Procura nas execuções anteriores
       for run_dir in run_dirs:
           for matrix_type, pattern in matrix_files.items():
               if matrix_type not in matrix_paths:
                   files = list(run_dir.glob(pattern))
                   if files:
                       latest = max(files, key=lambda x: x.stat().st_mtime)
                       matrix_paths[matrix_type] = latest
                       
           if len(matrix_paths) == 4:
               break
               
   return matrix_paths

def setup_run_dir() -> Path:
   """Cria e prepara novo diretório de execução"""
   i = 1
   while True:
       run_dir = EVAL_DIR / f"run_{i}"
       if not run_dir.exists():
           run_dir.mkdir(parents=True)
           
           # Cria subdiretórios
           (run_dir / "alignments").mkdir()
           (run_dir / "scores").mkdir()
           return run_dir
       i += 1

def copy_matrices_to_run(run_dir: Path) -> Dict[str, Path]:
   """Copia matrizes para o diretório da execução"""
   matrix_paths = get_latest_matrices()
   if not matrix_paths:
       logger.error("No matrices found")
       return {}
       
   new_paths = {}
   for matrix_type, src_path in matrix_paths.items():
       try:
           new_path = run_dir / src_path.name
           shutil.copy2(src_path, new_path)
           new_paths[matrix_type] = new_path
           logger.info(f"Copied {matrix_type} matrix to {run_dir}")
       except Exception as e:
           logger.error(f"Error copying {matrix_type} matrix: {e}")
           
   return new_paths

def run_clustalw(instance: str, matrix_path: Path, output_dir: Path) -> Optional[Path]:
   """Executa ClustalW com uma matriz específica"""
   input_file = BALIBASE_DIR / f"{instance}.tfa"
   temp_output = TEMP_DIR / f"{instance}_{matrix_path.stem}_temp.fasta"
   final_output = output_dir / f"{instance}_{matrix_path.stem}.aln"
   
   if not input_file.exists():
       logger.error(f"Input file not found: {input_file}")
       return None
   
   # Remove arquivo temporário se existir
   if temp_output.exists():
       temp_output.unlink()
   
   cmd = [
       str(CLUSTALW_PATH),
       f'-INFILE={str(input_file)}',
       f'-MATRIX={str(matrix_path)}',
       f'-OUTFILE={str(temp_output)}',
       '-OUTPUT=FASTA',
       '-TYPE=PROTEIN'
   ]
   
   try:
       subprocess.run(cmd, check=True, capture_output=True, text=True)
       
       if temp_output.exists():
           # Move para localização final
           shutil.move(str(temp_output), str(final_output))
           return final_output
           
       logger.error(f"ClustalW failed to generate output for {instance}")
       return None
       
   except subprocess.CalledProcessError as e:
       logger.error(f"ClustalW error for {instance}: {e.stderr}")
       return None

def get_bali_score(instance: str, alignment_path: Path) -> float:
   """Calcula BaliScore para um alinhamento"""
   xml_file = BALIBASE_DIR / f"{instance}.xml"
   
   if not xml_file.exists():
       logger.error(f"XML file not found: {xml_file}")
       return 0.0
   
   cmd = [str(BALISCORE_PATH), str(xml_file), str(alignment_path)]
   
   try:
       result = subprocess.run(cmd, check=True, capture_output=True, text=True)
       match = re.search(r'CS\s+score=\s*(\d+\.\d+)', result.stdout)
       if match:
           return float(match.group(1))
       logger.error(f"Could not find score in bali_score output for {instance}")
       return 0.0
   except subprocess.CalledProcessError as e:
       logger.error(f"BaliScore error for {instance}: {e.stderr}")
       return 0.0

def process_instance(
   instance: str, 
   run_dir: Path, 
   matrix_paths: Dict[str, Path]
) -> Dict:
   """Processa uma instância com todas as matrizes"""
   logger.info(f"Processing {instance}")
   
   alignments_dir = run_dir / "alignments" / instance
   alignments_dir.mkdir(exist_ok=True, parents=True)
   
   results = {}
   for matrix_type, matrix_path in matrix_paths.items():
       # Gera alinhamento
       alignment = run_clustalw(instance, matrix_path, alignments_dir)
       if alignment and alignment.exists():
           # Calcula score
           score = get_bali_score(instance, alignment)
           results[matrix_type] = score
           logger.info(f"{matrix_type} matrix score: {score:.4f}")
   
   if not results:
       logger.error(f"No successful alignments for {instance}")
       return {
           'instance': instance,
           'success': False
       }
   
   # Identifica melhor resultado
   best_type = max(results.items(), key=lambda x: x[1])[0]
   best_score = results[best_type]
   
   # Salva resumo da instância
   summary = {
       'instance': instance,
       'success': True,
       'best_matrix': best_type,
       'best_score': best_score,
       'scores': results
   }
   
   with open(alignments_dir / "summary.txt", "w") as f:
       f.write(f"Instance: {instance}\n")
       f.write(f"Best matrix type: {best_type}\n")
       f.write(f"Best score: {best_score:.4f}\n\n")
       f.write("All scores:\n")
       for mtype, score in results.items():
           f.write(f"{mtype}: {score:.4f}\n")
           
   return summary

def generate_final_report(run_dir: Path, all_results: List[Dict]) -> None:
   """Gera relatório final da execução e CSV com melhores scores"""
   matrix_stats = defaultdict(list)
   total_instances = len([r for r in all_results if r['success']])
   
   # Gera relatório detalhado
   report_path = run_dir / "final_report.txt"
   csv_path = run_dir / "best_scores.csv"
   
   # Gera relatório texto
   with open(report_path, "w") as f:
       f.write("=== MA-BioFit Evaluation Report ===\n\n")
       f.write(f"Total instances processed: {total_instances}\n")
       f.write(f"Run directory: {run_dir}\n")
       f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
       
       for result in all_results:
           if result['success']:
               for matrix_type, score in result['scores'].items():
                   matrix_stats[matrix_type].append(score)
       
       f.write("=== Matrix Performance ===\n")
       for matrix_type, scores in matrix_stats.items():
           if scores:
               wins = sum(1 for r in all_results if r.get('success') and 
                         r.get('best_matrix') == matrix_type)
               f.write(f"\n{matrix_type} Matrix:\n")
               f.write(f"  Times best: {wins}\n")
               f.write(f"  Average score: {sum(scores)/len(scores):.4f}\n")
               f.write(f"  Best score: {max(scores):.4f}\n")
               f.write(f"  Worst score: {min(scores):.4f}\n")
       
       f.write("\n=== Instance Details ===\n")
       for result in all_results:
           if result['success']:
               f.write(f"\n{result['instance']}:\n")
               f.write(f"  Best matrix: {result['best_matrix']}\n")
               for matrix_type, score in result['scores'].items():
                   f.write(f"  {matrix_type}: {score:.4f}\n")
                   
   # Gera CSV com melhores scores
   with open(csv_path, 'w', newline='') as f:
       writer = csv.writer(f, delimiter=';')
       writer.writerow(['Instance', 'Best_Score'])
       for result in all_results:
           if result['success']:
               writer.writerow([result['instance'], f"{result['best_score']:.4f}"])

def main():
   # Preparar diretório para nova execução
   run_dir = setup_run_dir()
   logger.info(f"Starting new evaluation run in {run_dir}")
   
   # Copiar matrizes
   matrix_paths = copy_matrices_to_run(run_dir)
   if not matrix_paths:
       logger.error("No matrices found to evaluate")
       return
   
   # Processar instâncias
   all_results = []
   for instance in INSTANCES:
       try:
           result = process_instance(instance, run_dir, matrix_paths)
           all_results.append(result)
       except Exception as e:
           logger.error(f"Error processing {instance}: {str(e)}")
           all_results.append({
               'instance': instance,
               'success': False,
               'error': str(e)
           })
   
   # Gerar relatório final
   generate_final_report(run_dir, all_results)
   logger.info(f"Evaluation complete. Results in {run_dir}")
   
   # Limpar arquivos temporários
   for f in TEMP_DIR.glob('*'):
       try:
           f.unlink()
       except Exception as e:
           logger.warning(f"Could not remove temp file {f}: {e}")

if __name__ == "__main__":
   main()