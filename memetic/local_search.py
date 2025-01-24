# memetic/local_search.py

import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict
import logging
import random
from datetime import datetime
from .matrix import AdaptiveMatrix


class LocalSearch:
    """
    VNS-ILS para otimização de matrizes adaptativas com mecanismos de escape.
    """

    def __init__(self, matrix: AdaptiveMatrix, hyperparams: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurações principais
        self.matrix = matrix
        self.hyperparams = hyperparams

        # Extrai parâmetros do hyperparams
        self.min_improvement = hyperparams['VNS']['MIN_IMPROVEMENT']
        self.perturbation_size = hyperparams['VNS']['PERTURBATION_SIZE']
        self.max_perturbation = hyperparams['VNS']['MAX_PERTURBATION']

        # Inicializações padrão
        self.best_matrix = matrix.copy()
        self.current_matrix = matrix.copy()
        self.best_score = float('-inf')

        # Estruturas de análise
        self.substitution_frequencies = defaultdict(lambda: defaultdict(float))
        self.conservation_weights = defaultdict(float)
        self.position_specific_patterns = defaultdict(list)

        # Tracking de progresso
        self.improvements = []
        self.stagnation_counter = 0
        self.cycle_detector = set()

        # Controle de temperatura para aceite de pioras
        self.current_temperature = 1.0
        self.cooling_rate = 0.95

        # Rastreamento de vizinhanças
        self.neighborhood_stats = {
            'frequency': {'attempts': 0, 'improvements': 0},
            'conservation': {'attempts': 0, 'improvements': 0},
            'group': {'attempts': 0, 'improvements': 0}
        }

        # Grupos físico-químicos dos aminoácidos
        self.aa_groups = {
            'hydrophobic': {'I', 'L', 'V', 'M', 'F', 'W', 'A'},
            'polar': {'S', 'T', 'N', 'Q'},
            'acidic': {'D', 'E'},
            'basic': {'K', 'R', 'H'},
            'special': {'C', 'G', 'P', 'Y'}
        }

    def analyze_alignment(self, xml_path: Path) -> None:
        """Analisa alinhamento para extrair padrões de substituição."""
        self.logger.info(f"Analyzing alignment: {xml_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            sequences = {}
            blocks = []

            # Extrai sequências e blocos
            for seq in root.findall(".//sequence"):
                seq_name = seq.find("seq-name").text
                seq_data = seq.find("seq-data").text.strip()
                sequences[seq_name] = seq_data

                for block in seq.findall(".//fitem/[ftype='BLOCK']"):
                    start = int(block.find("fstart").text)
                    stop = int(block.find("fstop").text)
                    score = float(block.find("fscore").text)

                    blocks.append({
                        'seq': seq_name,
                        'start': start,
                        'stop': stop,
                        'score': score,
                        'data': seq_data[start - 1:stop]
                    })

            self._analyze_patterns(sequences, blocks)

        except Exception as e:
            self.logger.error(f"Error analyzing alignment: {e}")
            raise

    def _analyze_patterns(self, sequences: Dict[str, str], blocks: List[Dict]) -> None:
        """Analisa padrões de substituição e conservação nos blocos."""
        self.logger.info("Analyzing substitution patterns")

        # Reset estruturas
        self.substitution_frequencies.clear()
        self.conservation_weights.clear()
        self.position_specific_patterns.clear()

        # Agrupa blocos por posição
        aligned_blocks = defaultdict(list)
        for block in blocks:
            aligned_blocks[block['start']].append(block)

        # Analisa cada posição alinhada
        for pos, block_group in aligned_blocks.items():
            conservation_score = np.mean([b['score'] for b in block_group])
            frequency_factor = len(block_group) / len(sequences)

            # Analisa substituições no bloco
            for i, block1 in enumerate(block_group):
                for j, block2 in enumerate(block_group[i + 1:], i + 1):
                    for offset in range(min(len(block1['data']), len(block2['data']))):
                        aa1 = block1['data'][offset]
                        aa2 = block2['data'][offset]

                        if aa1 != '-' and aa2 != '-':
                            # Peso considera conservação e frequência
                            weight = conservation_score * frequency_factor

                            # Ajusta peso baseado em contexto
                            if self._are_similar(aa1, aa2):
                                weight *= 1.2  # Bonus para substituições similares

                            self.substitution_frequencies[aa1][aa2] += weight
                            self.conservation_weights[aa1] += weight
                            self.conservation_weights[aa2] += weight

                            # Registra padrão específico da posição
                            self.position_specific_patterns[pos].append(
                                (aa1, aa2, weight)
                            )

        self.logger.info("Pattern analysis completed")

    def _are_similar(self, aa1: str, aa2: str) -> bool:
        """Verifica se dois aminoácidos pertencem ao mesmo grupo."""
        return any(aa1 in group and aa2 in group for group in self.aa_groups.values())

    def perturb_solution(self) -> AdaptiveMatrix:
        """Aplica perturbação adaptativa na solução atual."""
        self.logger.debug(f"Applying perturbation (size={self.perturbation_size})")

        perturbed = self.current_matrix.copy()
        positions = list(range(len(self.matrix.aa_order)))

        # Perturbação adaptativa baseada em estagnação
        num_changes = int(self.perturbation_size * (1 + self.stagnation_counter / 10))
        num_changes = min(num_changes, self.max_perturbation)

        changes_made = 0
        attempts = 0
        max_attempts = num_changes * 3

        while changes_made < num_changes and attempts < max_attempts:
            attempts += 1
            i, j = random.sample(positions, 2)
            aa1, aa2 = self.matrix.aa_order[i], self.matrix.aa_order[j]

            # Define magnitude da perturbação
            if self._are_similar(aa1, aa2):
                magnitude = random.choice([-1, 1])
            else:
                magnitude = random.choice([-2, -1, 1, 2])

            # Maior probabilidade de aumentar scores baixos
            current = perturbed.get_score(aa1, aa2)
            if current < 0 and random.random() < 0.7:
                magnitude = abs(magnitude)

            new_score = current + magnitude

            # Tenta aplicar mudança
            if perturbed._validate_score(aa1, aa2, new_score):
                perturbed.update_score(aa1, aa2, new_score)
                changes_made += 1

        return perturbed

    def _frequency_based_neighborhood(self) -> AdaptiveMatrix:
        """Vizinhança baseada em frequências de substituição."""
        self.logger.debug("Exploring frequency-based neighborhood")
        self.neighborhood_stats['frequency']['attempts'] += 1

        new_matrix = self.current_matrix.copy()

        # Seleciona substituições mais frequentes
        sorted_subs = [
            (aa1, aa2, freq)
            for aa1, subs in self.substitution_frequencies.items()
            for aa2, freq in subs.items()
        ]
        sorted_subs.sort(key=lambda x: x[2], reverse=True)

        # Modifica top substituições
        improvements = 0
        for aa1, aa2, freq in sorted_subs[:5]:
            current = new_matrix.get_score(aa1, aa2)

            # Ajuste proporcional à frequência
            median_freq = np.median([x[2] for x in sorted_subs]) if sorted_subs else 1
            direction = 1 if freq > median_freq else -1

            # Adiciona componente aleatório
            if random.random() < 0.3:
                direction *= -1

            new_score = current + direction

            # Aqui está a mudança: usar _validate_score em vez de _validate_score_change
            if new_matrix._validate_score(aa1, aa2, new_score):
                new_matrix.update_score(aa1, aa2, new_score)
                improvements += 1

        if improvements > 0:
            self.neighborhood_stats['frequency']['improvements'] += 1

        return new_matrix

    def _conservation_based_neighborhood(self) -> AdaptiveMatrix:
        """Vizinhança baseada em padrões de conservação."""
        self.logger.debug("Exploring conservation-based neighborhood")
        self.neighborhood_stats['conservation']['attempts'] += 1

        new_matrix = self.current_matrix.copy()

        # Seleciona aminoácidos mais conservados
        conserved = sorted(
            self.conservation_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:4]

        improvements = 0
        for aa1, weight1 in conserved:
            for aa2, weight2 in conserved:
                if aa1 != aa2:
                    current = new_matrix.get_score(aa1, aa2)

                    # Ajuste baseado em pesos de conservação
                    if weight1 > np.median([w for _, w in conserved]):
                        new_score = current + 1
                        if new_matrix._validate_score(aa1, aa2, new_score):
                            new_matrix.update_score(aa1, aa2, new_score)
                            improvements += 1

        if improvements > 0:
            self.neighborhood_stats['conservation']['improvements'] += 1

        return new_matrix

    def _group_based_neighborhood(self) -> AdaptiveMatrix:
        """Vizinhança baseada em grupos físico-químicos."""
        self.logger.debug("Exploring group-based neighborhood")
        self.neighborhood_stats['group']['attempts'] += 1

        new_matrix = self.current_matrix.copy()

        # Seleciona grupo aleatório
        group_aas = random.choice(list(self.aa_groups.values()))
        group_list = list(group_aas)

        improvements = 0

        # Fortalece relações dentro do grupo
        for i, aa1 in enumerate(group_list):
            for aa2 in group_list[i + 1:]:
                current = new_matrix.get_score(aa1, aa2)

                # Probabilidade de ajuste inversamente proporcional ao score atual
                if current < 5 and random.random() < 0.7:
                    new_score = current + 1
                    if new_matrix._validate_score(aa1, aa2, new_score):
                        new_matrix.update_score(aa1, aa2, new_score)
                        improvements += 1

        if improvements > 0:
            self.neighborhood_stats['group']['improvements'] += 1

        return new_matrix

    def accept_solution(self, new_score: float, current_score: float) -> bool:
        """Critério de aceite com temperatura adaptativa."""
        if new_score >= current_score:
            return True

        # Aceite probabilístico de pioras
        delta = new_score - current_score
        probability = np.exp(delta / self.current_temperature)

        # Temperatura diminui com estagnação
        self.current_temperature *= self.cooling_rate

        return random.random() < probability

    def vns_search(self, evaluation_func, max_iterations: int, max_no_improve: int) -> float:
        """VNS com ILS integrado e mecanismos adaptativos."""
        self.logger.info(f"Starting VNS-ILS search (max_iter={max_iterations})")

        current_score = evaluation_func(self.current_matrix)
        self.best_score = current_score
        iterations_no_improve = 0

        neighborhoods = [
            self._frequency_based_neighborhood,
            self._conservation_based_neighborhood,
            self._group_based_neighborhood
        ]

        while iterations_no_improve < max_no_improve:
            self.logger.debug(f"Iteration {iterations_no_improve + 1}/{max_no_improve}")

            # Aplica perturbação em caso de estagnação
            if iterations_no_improve > max_no_improve // 2:
                self.stagnation_counter += 1
                self.current_matrix = self.perturb_solution()
                current_score = evaluation_func(self.current_matrix)

                # Reinicia temperatura
                self.current_temperature = 1.0

                self.logger.debug(
                    f"Applied perturbation (stagnation={self.stagnation_counter}). "
                    f"New score: {current_score:.4f}"
                )

            # Exploração de vizinhanças
            improved = False
            for k, neighborhood in enumerate(neighborhoods):
                neighbor = neighborhood()
                neighbor_score = evaluation_func(neighbor)

                # Detecta ciclos
                neighbor_hash = hash(str(neighbor.matrix))
                if neighbor_hash in self.cycle_detector:
                    continue

                self.cycle_detector.add(neighbor_hash)
                if len(self.cycle_detector) > 1000:  # Limita tamanho
                    self.cycle_detector.clear()

                # Critério de aceite
                if self.accept_solution(neighbor_score, current_score):
                    improvement = neighbor_score - current_score
                    self.current_matrix = neighbor
                    current_score = neighbor_score

                    if improvement > self.min_improvement:
                        self.improvements.append({
                            'score_improvement': improvement,
                            'neighborhood': k,
                            'iteration': iterations_no_improve
                        })
                        improved = True
                        break

            # Atualiza melhor solução
            if current_score > self.best_score + self.min_improvement:
                self.best_matrix = self.current_matrix.copy()
                self.best_score = current_score
                iterations_no_improve = 0
                self.stagnation_counter = 0
                self.perturbation_size = self.hyperparams['VNS']['PERTURBATION_SIZE']
                self.logger.info(f"New best score found: {self.best_score:.4f}")
            else:
                iterations_no_improve += 1

                # Ajusta tamanho da perturbação
                if iterations_no_improve > max_no_improve // 2:
                    self.perturbation_size = min(
                        self.max_perturbation,
                        self.perturbation_size + 1
                    )

            # Log do progresso
            if iterations_no_improve % 5 == 0:
                success_rates = {
                    name: stats['improvements'] / max(1, stats['attempts'])
                    for name, stats in self.neighborhood_stats.items()
                }

                self.logger.debug(
                    f"Progress: score={current_score:.4f}, "
                    f"best={self.best_score:.4f}, "
                    f"stagnation={iterations_no_improve}, "
                    f"perturbation={self.perturbation_size}, "
                    f"temperature={self.current_temperature:.2f}, "
                    f"neighborhood_success={success_rates}"
                )

        # Restaura melhor solução encontrada
        self.current_matrix = self.best_matrix.copy()
        self.logger.info(f"VNS-ILS search completed. Best score: {self.best_score:.4f}")

        # Análise final das vizinhanças
        total_improvements = sum(
            stats['improvements']
            for stats in self.neighborhood_stats.values()
        )

        if total_improvements > 0:
            for name, stats in self.neighborhood_stats.items():
                success_rate = stats['improvements'] / max(1, stats['attempts'])
                contribution = stats['improvements'] / total_improvements
                self.logger.info(
                    f"Neighborhood {name}: "
                    f"success_rate={success_rate:.2%}, "
                    f"contribution={contribution:.2%}"
                )

        return self.best_score

    def save_best_matrix(self, results_dir: Path, run_info: str) -> None:
        """Salva melhor matriz encontrada com metadados."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            filename = f"{timestamp}-AdaptivePAM-{run_info}.txt"
            output_path = results_dir / filename

            # Salva matriz
            self.best_matrix.to_clustalw_format(output_path)

            # Salva metadados
            meta_path = output_path.with_suffix('.meta.json')
            meta_data = {
                'timestamp': timestamp,
                'score': self.best_score,
                'improvements': self.improvements,
                'neighborhood_stats': self.neighborhood_stats,
                'hyperparams': self.hyperparams
            }

            with open(meta_path, 'w') as f:
                import json
                json.dump(meta_data, f, indent=2)

            self.logger.info(f"Results saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
