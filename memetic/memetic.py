import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional, Callable
import random
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

from .matrix import AdaptiveMatrix
from .local_search import EnhancedLocalSearch


@dataclass
class Individual:
    matrix: AdaptiveMatrix
    fitness: float = float('-inf')
    local_search_count: int = 0

    def copy(self) -> 'Individual':
        return Individual(
            matrix=self.matrix.copy(),
            fitness=self.fitness,
            local_search_count=self.local_search_count
        )


class StructuredPopulation:
    """
    População estruturada hierarquicamente com conhecimento de blocos conservados
    e regiões SEQERR do BAliBASE4.
    """

    def __init__(
        self,
        evaluation_function: Callable,
        local_search: EnhancedLocalSearch,
        hyperparams: Dict,
        xml_path: Optional[Path] = None
    ):
        self.evaluate = evaluation_function
        self.local_search = local_search
        self.hyperparams = hyperparams
        self.logger = logging.getLogger(self.__class__.__name__)

        # Estrutura hierárquica da população
        self.hierarchy = {
            'master': 0,
            'subordinates': [1, 2, 3],
            'workers': [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
        }

        # Análise do XML para guiar operadores
        self.block_scores = {}  # Scores dos blocos conservados
        self.sequence_groups = {}  # Grupos de sequências
        self.seqerr_regions = set()  # Regiões com erros/discrepâncias
        if xml_path:
            self._analyze_balibase_data(xml_path)

        self.individuals = []
        self._initialize_population()

    def _analyze_balibase_data(self, xml_path: Path) -> None:
        """
        Analisa dados do BAliBASE4 para guiar os operadores genéticos.
        Extrai informações sobre blocos conservados, grupos e regiões SEQERR.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extrai scores dos blocos
            for block in root.findall(".//fitem/[ftype='BLOCK']"):
                start = int(block.find("fstart").text)
                stop = int(block.find("fstop").text)
                score = float(block.find("fscore").text)
                self.block_scores[(start, stop)] = score

            # Mapeia grupos de sequências
            for seq in root.findall(".//sequence"):
                seq_name = seq.find("seq-name").text
                group = int(seq.find("seq-info/group").text)
                self.sequence_groups[seq_name] = group

            # Identifica regiões SEQERR baseado em:
            # - Blocos com scores muito baixos
            # - Gaps inconsistentes entre grupos
            # - Discrepâncias estruturais
            threshold = self.hyperparams.get('SEQERR_SCORE_THRESHOLD', 10.0)
            for (start, stop), score in self.block_scores.items():
                if score < threshold:
                    self.seqerr_regions.add((start, stop))

            self.logger.info(f"Analyzed BAliBASE4 data: {len(self.block_scores)} blocks, "
                             f"{len(self.seqerr_regions)} SEQERR regions")

        except Exception as e:
            self.logger.error(f"Error analyzing BAliBASE4 data: {e}")
            raise

    def _initialize_population(self) -> None:
        """
        Inicializa população com variações da PAM250 original.
        Aplica mudanças mais conservadoras em regiões bem definidas.
        """
        self.individuals = [Individual(AdaptiveMatrix(self.hyperparams)) for _ in range(13)]

        for ind in self.individuals[1:]:
            modified = False
            for i, aa1 in enumerate(ind.matrix.aa_order):
                for j, aa2 in enumerate(ind.matrix.aa_order[i:], i):
                    # Probabilidade menor de modificar scores altos na diagonal
                    if aa1 == aa2:
                        prob = 0.1
                    else:
                        prob = 0.3

                    if random.random() < prob:
                        current = ind.matrix.get_score(aa1, aa2)
                        adjustment = self._get_initial_adjustment(aa1, aa2)
                        new_score = current + adjustment
                        if ind.matrix._validate_score(aa1, aa2, new_score):
                            ind.matrix.update_score(aa1, aa2, new_score)
                            modified = True

            if not modified:
                self._generate_random_changes(ind.matrix)

        self.evaluate_population()

    def _get_initial_adjustment(self, aa1: str, aa2: str) -> int:
        """
        Define ajustes iniciais baseados no tipo de par de aminoácidos.
        Considera estrutura química e dados do BAliBASE4.
        """
        if aa1 == aa2:  # Diagonal
            return random.choice([-2, -1])  # Mais conservador
        elif any(aa1 in group and aa2 in group
                 for group in self.local_search.matrix.similar_groups):
            return random.choice([-2, -1, 1])  # Flexível para similares
        else:
            return random.choice([-2, -1])  # Conservador para diferentes

    def _generate_random_changes(self, matrix: AdaptiveMatrix, num_changes: int = 5) -> None:
        """Gera mudanças aleatórias respeitando restrições biológicas."""
        for _ in range(num_changes):
            aa1 = random.choice(matrix.aa_order)
            aa2 = random.choice(matrix.aa_order)
            current = matrix.get_score(aa1, aa2)

            # Ajustes mais conservadores
            if aa1 == aa2:
                adjustment = random.choice([-1, 1])
            else:
                adjustment = random.choice([-2, -1, 1])

            new_score = current + adjustment
            if matrix._validate_score(aa1, aa2, new_score):
                matrix.update_score(aa1, aa2, new_score)

    def evaluate_population(self) -> None:
        """Avalia população e mantém registro do melhor global."""
        for ind in self.individuals:
            if ind.fitness == float('-inf'):
                ind.fitness = self.evaluate(ind.matrix)

        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def hierarchical_crossover(self) -> None:
        """
        Crossover hierárquico considerando estrutura de blocos e grupos.
        Preserva características importantes identificadas no BAliBASE4.
        """
        new_individuals = []

        # Crossover entre subordinados e trabalhadores
        for sub_idx, worker_group in zip(
            self.hierarchy['subordinates'],
            self.hierarchy['workers']
        ):
            parent = self.individuals[sub_idx]
            for worker_idx in worker_group:
                child = self._informed_crossover(parent, self.individuals[worker_idx])
                mutant = self._structural_mutation(child)

                if mutant.fitness > child.fitness:
                    child = mutant

                if child.fitness > self.individuals[worker_idx].fitness:
                    new_individuals.append((worker_idx, child))

        # Crossover entre mestre e subordinados
        master = self.individuals[self.hierarchy['master']]
        for sub_idx in self.hierarchy['subordinates']:
            child = self._informed_crossover(master, self.individuals[sub_idx])
            mutant = self._structural_mutation(child)

            if mutant.fitness > child.fitness:
                child = mutant

            if child.fitness > self.individuals[sub_idx].fitness:
                new_individuals.append((sub_idx, child))

        # Atualiza população
        for idx, child in new_individuals:
            self.individuals[idx] = child

        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def _informed_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Crossover informado por dados do BAliBASE4.
        Considera blocos conservados, grupos e scores ao herdar características.
        """
        child = Individual(AdaptiveMatrix(self.hyperparams))

        # Verifica se diagonais são diferentes antes de herdar
        diagonals_differ = False
        for aa in child.matrix.aa_order:
            score1 = parent1.matrix.get_score(aa, aa)
            score2 = parent2.matrix.get_score(aa, aa)
            if score1 != score2:
                diagonals_differ = True
                break

        # Se diagonais diferem, escolhe baseado em fitness
        if diagonals_differ:
            better_parent = parent1 if parent1.fitness > parent2.fitness else parent2
            for aa in child.matrix.aa_order:
                diag_score = better_parent.matrix.get_score(aa, aa)
                child.matrix.update_score(aa, aa, diag_score)
        else:
            # Se iguais, herda diagonais de qualquer um
            for aa in child.matrix.aa_order:
                diag_score = parent1.matrix.get_score(aa, aa)
                child.matrix.update_score(aa, aa, diag_score)

        # Para outros elementos, considera estrutura dos dados
        for i, aa1 in enumerate(child.matrix.aa_order):
            for j, aa2 in enumerate(child.matrix.aa_order[i + 1:], i + 1):
                # Determina probabilidade baseada em blocos/grupos
                inherit_prob = self._calculate_inheritance_probability(aa1, aa2)

                if random.random() < inherit_prob:
                    score = parent1.matrix.get_score(aa1, aa2)
                else:
                    score = parent2.matrix.get_score(aa1, aa2)

                if child.matrix._validate_score(aa1, aa2, score):
                    child.matrix.update_score(aa1, aa2, score)

        child.fitness = self.evaluate(child.matrix)
        return child

    def _calculate_inheritance_probability(self, aa1: str, aa2: str) -> float:
        """
        Calcula probabilidade de herança baseada em dados estruturais.
        Considera grupos físico-químicos e padrões do BAliBASE4.
        """
        base_prob = 0.5

        # Ajusta probabilidade baseado em grupos físico-químicos
        if any(aa1 in group and aa2 in group
               for group in self.local_search.matrix.similar_groups):
            base_prob += 0.1

        # Ajusta baseado em blocos conservados
        if any((start, stop) in self.block_scores
               for (start, stop), score in self.block_scores.items()
               if score > 20):
            base_prob += 0.1

        # Reduz probabilidade em regiões SEQERR
        if any((start, stop) in self.seqerr_regions
               for (start, stop) in self.seqerr_regions):
            base_prob -= 0.1

        return min(max(base_prob, 0.3), 0.7)  # Mantém entre 0.3 e 0.7

    def _structural_mutation(self, individual: Individual) -> Individual:
        """
        Mutação estrutural que considera regiões SEQERR e conservação.
        Aplica mudanças mais fortes em regiões problemáticas.
        """
        mutant = individual.copy()
        mutation_rate = self.hyperparams['MEMETIC']['MUTATION_RATE']

        for i, aa1 in enumerate(mutant.matrix.aa_order):
            for j, aa2 in enumerate(mutant.matrix.aa_order[i:], i):
                if random.random() < mutation_rate:
                    current = mutant.matrix.get_score(aa1, aa2)

                    # Aumenta força de mutações negativas em regiões SEQERR
                    in_seqerr = any((start, stop) in self.seqerr_regions
                                   for (start, stop) in self.seqerr_regions)

                    if in_seqerr and current < 0:
                        # 50% mais forte para mutações negativas em SEQERR
                        adjustment = random.choice([-3, -2])
                    else:
                        # Mutação normal para outros casos
                        adjustment = random.choice([-2, -1, 1])

                    new_score = current + adjustment
                    if mutant.matrix._validate_score(aa1, aa2, new_score):
                        mutant.matrix.update_score(aa1, aa2, new_score)

        mutant.fitness = self.evaluate(mutant.matrix)
        return mutant


class MemeticAlgorithm:
    """
    Algoritmo Memético aprimorado para otimização de matrizes PAM
    considerando estrutura do BAliBASE4.
    """

    def __init__(
        self,
        evaluation_function: Callable,
        xml_path: Path,
        hyperparams: Dict
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hyperparams = hyperparams
        self.evaluation_function = evaluation_function

        # Inicializa busca local com análise do BAliBASE4
        self.local_search = EnhancedLocalSearch(
            matrix=AdaptiveMatrix(hyperparams),
            hyperparams=hyperparams
        )

        if xml_path:
            self.local_search.analyze_alignment(xml_path)

        # Inicializa população estruturada
        self.population = StructuredPopulation(
            evaluation_function=evaluation_function,
            local_search=self.local_search,
            hyperparams=hyperparams,
            xml_path=xml_path
        )

        self.best_global_matrix = None
        self.best_global_score = float('-inf')
        self.initial_score = None

        self.initialize_population()

    def initialize_population(self):
        """Inicializa população e registra estado inicial."""
        self.population.evaluate_population()
        best_individual = self.population.individuals[0]

        self.best_global_matrix = best_individual.matrix.copy()
        self.best_global_score = best_individual.fitness
        self.initial_score = best_individual.fitness
        self.logger.info(f"Initial best fitness: {self.initial_score:.4f}")

    def run(
        self,
        generations: int,
        local_search_frequency: int,
        local_search_iterations: int,
        max_no_improve: int,
        evaluation_function: Callable
    ) -> Tuple[AdaptiveMatrix, float]:
        """
        Executa otimização memética com controle de convergência e tracking de melhores soluções.

        Args:
            generations: Número máximo de gerações
            local_search_frequency: Frequência de aplicação da busca local
            local_search_iterations: Iterações máximas da busca local
            max_no_improve: Limite de gerações sem melhoria
            evaluation_function: Função de avaliação

        Returns:
            Tuple contendo melhor matriz encontrada e seu score
        """
        self.logger.info("Starting memetic optimization with enhanced operators")
        stagnation_counter = 0
        generation_scores = []  # Tracking de scores por geração

        for generation in range(generations):
            self.population.evaluate_population()

            if generation % local_search_frequency == 0:
                self._apply_local_search(
                    local_search_iterations,
                    max_no_improve,
                    evaluation_function
                )

            self.population.hierarchical_crossover()

            # Avalia e atualiza melhor global
            current_best = self.population.individuals[0]
            current_score = evaluation_function(current_best.matrix)
            generation_scores.append(current_score)

            if current_score > self.best_global_score:
                self.best_global_score = current_score
                self.best_global_matrix = current_best.matrix.copy()
                stagnation_counter = 0
                self.logger.info(f"New best score in generation {generation}: {self.best_global_score:.4f}")
            else:
                stagnation_counter += 1

            # Critério de parada por estagnação
            if stagnation_counter >= max_no_improve:
                self.logger.info(
                    f"Stopping early at generation {generation} due to stagnation. "
                    f"Best score: {self.best_global_score:.4f}"
                )
                break

            # Log periódico
            if generation % 5 == 0:
                self.logger.info(
                    f"Generation {generation}: "
                    f"Best={self.best_global_score:.4f}, "
                    f"Current={current_score:.4f}, "
                    f"Stagnation={stagnation_counter}"
                )

        # Validação final e retorno
        if self.best_global_matrix is not None:
            final_score = evaluation_function(self.best_global_matrix)

            # Garante que retornamos o verdadeiro melhor encontrado
            if final_score > self.best_global_score:
                self.best_global_score = final_score

            self.logger.info(
                f"Optimization completed. Initial score: {self.initial_score:.4f}, "
                f"Final best: {self.best_global_score:.4f}"
            )

            return self.best_global_matrix.copy(), self.best_global_score

        return None, float('-inf')

    def _apply_local_search(
        self,
        local_search_iterations: int,
        max_no_improve: int,
        evaluation_function: Callable
    ) -> None:
        """
        Aplica busca local nos indivíduos superiores da hierarquia.
        Garante que melhorias são adequadamente propagadas e registradas.
        """
        self.logger.info("Applying enhanced local search with structural neighborhoods")

        # Aplica nos indivíduos superiores da hierarquia
        for idx in [self.population.hierarchy['master']] + self.population.hierarchy['subordinates']:
            individual = self.population.individuals[idx]

            # Limita número de aplicações por indivíduo
            if individual.local_search_count < 3:
                self.local_search.matrix = individual.matrix
                new_score = self.local_search.vns_search(
                    evaluation_func=evaluation_function,
                    max_iterations=local_search_iterations,
                    max_no_improve=max_no_improve
                )

                # Atualiza se houve melhoria
                if new_score > individual.fitness:
                    individual.fitness = new_score
                    individual.matrix = self.local_search.best_matrix.copy()

                    # Atualiza melhor global se necessário
                    if new_score > self.best_global_score:
                        validated_score = evaluation_function(self.local_search.best_matrix)
                        if validated_score > self.best_global_score:
                            self.best_global_score = validated_score
                            self.best_global_matrix = self.local_search.best_matrix.copy()
                            self.logger.info(f"New best score from local search: {self.best_global_score:.4f}")

                individual.local_search_count += 1

        # Reavalia população após busca local
        self.population.evaluate_population()
