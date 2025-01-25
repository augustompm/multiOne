# memetic/memetic.py

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable
import logging
from copy import deepcopy
import random
from datetime import datetime

from .matrix import AdaptiveMatrix
from .local_search import LocalSearch


class Individual:
    """
    Representa uma solução (matriz adaptativa) com seu histórico de otimização.
    """
    def __init__(self, matrix: Optional[AdaptiveMatrix] = None):
        self.matrix = matrix if matrix else AdaptiveMatrix()
        self.fitness = float('-inf')
        self.age = 0  # Controle de idade para diversidade
        self.parent_ids = set()  # Rastreia ancestrais
        self.last_improvement = 0  # Última geração com melhoria
        self.local_search_count = 0

    def copy(self) -> 'Individual':
        new_ind = Individual()
        new_ind.matrix = self.matrix.copy()
        new_ind.fitness = self.fitness
        new_ind.age = self.age
        new_ind.parent_ids = self.parent_ids.copy()
        new_ind.last_improvement = self.last_improvement
        new_ind.local_search_count = self.local_search_count
        return new_ind


class ElitePool:
    """
    Gerencia a elite com mecanismos para manter diversidade.
    """
    def __init__(self, size: int, diversity_threshold: float):
        self.size = size
        self.diversity_threshold = diversity_threshold
        self.individuals: List[Individual] = []
        self.history: Dict[int, float] = {}  # Histórico de fitness por geração

    def add(self, individual: Individual, generation: int) -> bool:
        """Tenta adicionar indivíduo mantendo diversidade."""
        # Registra histórico
        self.history[generation] = max(
            self.history.get(generation, float('-inf')),
            individual.fitness
        )

        # Pool não está cheio
        if len(self.individuals) < self.size:
            self.individuals.append(individual.copy())
            self._sort_pool()
            return True

        # Verifica se é melhor que o pior elite
        if individual.fitness <= self.individuals[-1].fitness:
            return False

        # Calcula distâncias para todos os elite
        distances = [
            self._calculate_distance(individual, elite)
            for elite in self.individuals
        ]

        # Se muito similar a algum elite existente
        min_distance = min(distances)
        if min_distance < self.diversity_threshold:
            similar_idx = distances.index(min_distance)
            # Substitui apenas se significativamente melhor
            if individual.fitness > self.individuals[similar_idx].fitness * 1.05:
                self.individuals[similar_idx] = individual.copy()
                self._sort_pool()
                return True
            return False

        # Adiciona novo indivíduo, remove pior
        self.individuals[-1] = individual.copy()
        self._sort_pool()
        return True

    def _calculate_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calcula distância entre matrizes considerando padrões de substituição."""
        diff_matrix = np.abs(ind1.matrix.matrix - ind2.matrix.matrix)

        # Pesos diferentes para diferentes regiões da matriz
        weights = np.ones_like(diff_matrix)
        n = len(ind1.matrix.aa_order)

        # Maior peso para diagonal
        for i in range(n):
            weights[i, i] = 2.0

        # Peso intermediário para aminoácidos similares
        for i in range(n):
            for j in range(i + 1, n):
                if ind1.matrix._are_similar(
                    ind1.matrix.aa_order[i],
                    ind1.matrix.aa_order[j]
                ):
                    weights[i, j] = weights[j, i] = 1.5

        return np.average(diff_matrix, weights=weights)

    def _sort_pool(self):
        """Ordena pool por fitness e idade."""
        self.individuals.sort(
            key=lambda x: (x.fitness, -x.age),  # Prioriza fitness, desempata por idade
            reverse=True
        )

    def get_diverse_parents(self, num_parents: int) -> List[Individual]:
        """Seleciona pais diversos para crossover."""
        if len(self.individuals) < num_parents:
            return self.individuals.copy()

        selected = []
        available = self.individuals.copy()

        while len(selected) < num_parents and available:
            if not selected:
                # Primeiro pai é o melhor disponível
                selected.append(available.pop(0))
            else:
                # Próximos pais são os mais distantes dos já selecionados
                distances = [
                    sum(self._calculate_distance(ind, sel) for sel in selected)
                    for ind in available
                ]
                idx = distances.index(max(distances))
                selected.append(available.pop(idx))

        return selected

    def update_ages(self):
        """Incrementa idade dos indivíduos."""
        for ind in self.individuals:
            ind.age += 1

    def get_stagnation_generations(self) -> int:
        """Retorna número de gerações sem melhoria significativa."""
        if len(self.history) < 2:
            return 0

        recent_best = max(
            v for k, v in self.history.items()
            if k >= max(0, max(self.history.keys()) - 10)
        )
        return sum(1 for v in self.history.values() if v >= recent_best * 0.99)


class Population:
    """
    Gerencia população com mecanismos adaptativos.
    """
    def __init__(
        self,
        size: int,
        elite_size: int,
        evaluation_function: Callable,
        local_search: LocalSearch,
        hyperparams: Dict
    ):
        self.size = size
        self.individuals: List[Individual] = []
        self.elite_pool = ElitePool(
            elite_size,
            hyperparams['MEMETIC']['DIVERSITY_THRESHOLD']
        )
        self.evaluate = evaluation_function
        self.local_search = local_search
        self.hyperparams = hyperparams

        # Inicializa população
        self._initialize_population()

    def _initialize_population(self):
        """Inicializa população com diversidade controlada."""
        # Primeiro indivíduo é PAM250 padrão
        self.individuals.append(Individual())

        # Cria variações do PAM250
        while len(self.individuals) < self.size:
            new_ind = Individual()

            # Aplica perturbações estruturadas
            for i, aa1 in enumerate(new_ind.matrix.aa_order):
                for j, aa2 in enumerate(new_ind.matrix.aa_order[i:], i):
                    if random.random() < 0.15:  # 15% chance de modificação
                        current = new_ind.matrix.get_score(aa1, aa2)

                        # Ajuste baseado em propriedades dos AAs
                        if new_ind.matrix._are_similar(aa1, aa2):
                            adjustment = random.choice([-1, 1])
                        else:
                            adjustment = random.choice([-2, -1, 1, 2])

                        # Alteração aplicada aqui: _validate_score em vez de _validate_score_change
                        if new_ind.matrix._validate_score(aa1, aa2, current + adjustment):
                            new_ind.matrix.update_score(aa1, aa2, current + adjustment)

            self.individuals.append(new_ind)

    def evaluate_population(self):
        """Avalia população garantindo precisão."""
        for ind in self.individuals:
            if ind.fitness == float('-inf'):
                # Múltiplas avaliações para reduzir ruído
                scores = []
                for _ in range(self.hyperparams['EXECUTION']['EVAL_SAMPLES']):
                    score = self.evaluate(ind.matrix)
                    if score > 0:
                        scores.append(score)

                ind.fitness = np.mean(scores) if scores else 0.0

    def create_next_generation(self, generation: int):
        """Cria próxima geração com mecanismos adaptativos."""
        new_population = []

        # Mantém elite
        for elite in self.elite_pool.individuals:
            new_population.append(elite.copy())

        # Detecta estagnação
        stagnation = self.elite_pool.get_stagnation_generations()

        # Ajusta parâmetros baseado em estagnação
        if stagnation > self.hyperparams['VNS']['ESCAPE_THRESHOLD']:
            perturbation_size = min(
                self.hyperparams['VNS']['MAX_PERTURBATION'],
                self.hyperparams['VNS']['PERTURBATION_SIZE'] * (1 + stagnation / 10)
            )
        else:
            perturbation_size = self.hyperparams['VNS']['PERTURBATION_SIZE']

        # Preenche resto da população
        while len(new_population) < self.size:
            if random.random() < 0.7:  # 70% chance de crossover
                # Seleciona pais diversos
                parents = self.elite_pool.get_diverse_parents(2)
                if len(parents) >= 2:
                    child = self._crossover(parents[0], parents[1])
                else:
                    child = self._mutate(self.elite_pool.individuals[0])
            else:
                # Mutação mais agressiva em caso de estagnação
                template = random.choice(self.elite_pool.individuals)
                child = self._mutate(
                    template,
                    mutation_rate=0.1 * (1 + stagnation / 10)
                )

            new_population.append(child)

        self.individuals = new_population
        self.elite_pool.update_ages()

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover adaptativo baseado em padrões de substituição."""
        child = Individual()

        # Herança baseada em fitness relativo
        total_fitness = parent1.fitness + parent2.fitness
        p1_weight = parent1.fitness / total_fitness if total_fitness != 0 else 0.5

        for i, aa1 in enumerate(child.matrix.aa_order):
            for j, aa2 in enumerate(child.matrix.aa_order[i:], i):
                if random.random() < p1_weight:
                    score = parent1.matrix.get_score(aa1, aa2)
                else:
                    score = parent2.matrix.get_score(aa1, aa2)

                # Pequena chance de inovação
                if random.random() < 0.1:
                    adjustment = random.choice([-1, 1])
                    score += adjustment

                # Alteração aplicada aqui: _validate_score em vez de _validate_score_change
                if child.matrix._validate_score(aa1, aa2, score):
                    child.matrix.update_score(aa1, aa2, score)

        # Registra parentesco
        child.parent_ids = {id(parent1), id(parent2)}
        return child

    def _mutate(self, template: Individual, mutation_rate: float = 0.1) -> Individual:
        """Mutação com taxa adaptativa."""
        child = template.copy()

        for i, aa1 in enumerate(child.matrix.aa_order):
            for j, aa2 in enumerate(child.matrix.aa_order[i:], i):
                if random.random() < mutation_rate:
                    current = child.matrix.get_score(aa1, aa2)

                    # Ajuste baseado em propriedades
                    if child.matrix._are_similar(aa1, aa2):
                        adjustment = random.choice([-1, 1])
                    else:
                        adjustment = random.choice([-2, -1, 1, 2])

                    # Alteração aplicada aqui: _validate_score em vez de _validate_score_change
                    if child.matrix._validate_score(
                        aa1, aa2, current + adjustment
                    ):
                        child.matrix.update_score(aa1, aa2, current + adjustment)

        return child


class MemeticAlgorithm:
    """
    Coordena o algoritmo memético com mecanismos adaptativos.
    """
    def __init__(
        self,
        population_size: int,
        elite_size: int,
        evaluation_function: Callable,
        xml_path: Path,
        max_generations: int,
        local_search_frequency: int,
        hyperparams: Dict
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hyperparams = hyperparams

        # Inicializa busca local
        self.local_search = LocalSearch(
            matrix=AdaptiveMatrix(),
            hyperparams=hyperparams  # Comentário removido para refletir a passagem correta dos parâmetros
        )

        if xml_path:
            self.local_search.analyze_alignment(xml_path)

        # Inicializa população
        self.population = Population(
            size=population_size,
            elite_size=elite_size,
            evaluation_function=evaluation_function,
            local_search=self.local_search,
            hyperparams=hyperparams
        )

        self.best_global_matrix = None
        self.best_global_score = float('-inf')
        self.initial_score = None

        # Inicializa população
        self.initialize_population()

    def initialize_population(self):
        """Inicializa população e registra score inicial."""
        self.population.evaluate_population()
        initial_best = self.population.elite_pool.individuals[0] if self.population.elite_pool.individuals else None

        if initial_best:
            self.best_global_matrix = initial_best.matrix.copy()
            self.best_global_score = initial_best.fitness
            self.initial_score = initial_best.fitness
            self.logger.info(f"Initial best fitness: {self.initial_score:.4f}")
        else:
            self.logger.warning("No viable individuals in initial population")

    def run(
        self,
        generations: int,
        local_search_frequency: int,
        local_search_iterations: int,
        max_no_improve: int,
        run_id: int,  # Adicionado run_id
        evaluation_function: Callable  # Adicionado evaluation_function
    ) -> AdaptiveMatrix:
        """Executa otimização com mecanismos adaptativos."""
        self.logger.info(f"Starting memetic optimization for Run {run_id}")
        best_of_this_run = float('-inf')
        best_matrix_of_run = None

        for generation in range(1, generations + 1):
            # Avalia população
            self.population.evaluate_population()

            # Aplicação da busca local em frequências definidas
            if generation % local_search_frequency == 0:
                # Ajusta intensidade da busca local baseado em estagnação
                stagnation = self.population.elite_pool.get_stagnation_generations()
                intensity = min(1.0, stagnation / self.hyperparams['VNS']['ESCAPE_THRESHOLD'])
                num_candidates = max(
                    1,
                    int(self.population.size * 0.2 * (1 + intensity))
                )

                # Seleciona candidatos para busca local
                candidates = sorted(
                    self.population.individuals,
                    key=lambda x: x.fitness,
                    reverse=True
                )[:num_candidates]

                for candidate in candidates:
                    if candidate.local_search_count < 3:  # Limita aplicações de busca local
                        self.local_search.matrix = candidate.matrix
                        new_score = self.local_search.vns_search(
                            evaluation_func=lambda m: self.population.evaluate(m),
                            max_iterations=int(local_search_iterations * (1 + intensity)),
                            max_no_improve=max_no_improve
                        )

                        if new_score > candidate.fitness:
                            candidate.fitness = new_score
                            candidate.last_improvement = generation

                        candidate.local_search_count += 1

            # Atualiza elite pool
            for ind in self.population.individuals:
                self.population.elite_pool.add(ind, generation)

                # Atualiza melhor global
                if ind.fitness > self.best_global_score:
                    self.best_global_score = ind.fitness
                    self.best_global_matrix = ind.matrix.copy()
                    self.logger.info(f"Run {run_id}: New best global score: {self.best_global_score:.4f}")

                    # Salva imediatamente a nova melhor matriz
                    try:
                        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
                        results_dir = Path("memetic/results")
                        results_dir.mkdir(exist_ok=True, parents=True)

                        matrix_file = results_dir / f"{timestamp}-Run{run_id}-AdaptivePAM-{self.best_global_score:.4f}.txt"
                        self.best_global_matrix.to_clustalw_format(matrix_file)
                        self.logger.info(f"Run {run_id}: Saved best matrix to: {matrix_file}")
                    except Exception as e:
                        self.logger.error(f"Run {run_id}: Error saving matrix: {e}")

            # Atualiza melhor desta execução
            current_best = max(self.population.individuals, key=lambda ind: ind.fitness, default=None)
            if current_best:
                # Valida com bali_score antes de considerar como melhor
                current_score = evaluation_function(current_best.matrix)
                if current_score > best_of_this_run:
                    best_of_this_run = current_score
                    best_matrix_of_run = current_best.matrix.copy()
                    self.logger.info(f"Run {run_id}: New best score (validated): {best_of_this_run:.4f}")

            # Verifica critério de parada antecipada
            stagnation = self.population.elite_pool.get_stagnation_generations()
            if stagnation > self.hyperparams['VNS']['MAX_NO_IMPROVE']:
                # Aplica reinício parcial se estagnado
                if random.random() < 0.3:  # 30% chance de reinício
                    self.logger.info(f"Run {run_id}: Applying partial restart due to stagnation")
                    self._partial_restart()
                    continue

                # Ou termina se já próximo do máximo de gerações
                if generation > generations * 0.8:
                    self.logger.info(f"Run {run_id}: Early stopping due to stagnation")
                    break

            # Cria próxima geração
            self.population.create_next_generation(generation)

            # Log do progresso
            if generation % 10 == 0:  # Log a cada 10 gerações
                elite_fitness = [ind.fitness for ind in self.population.elite_pool.individuals]
                elite_avg = np.mean(elite_fitness) if elite_fitness else 0.0
                self.logger.info(
                    f"Run {run_id} - Generation {generation}: "
                    f"Best={self.best_global_score:.4f}, "
                    f"Elite_avg={elite_avg:.4f}, "
                    f"Stagnation={stagnation}"
                )

        # Salva apenas se for o melhor global após validação
        if best_of_this_run > self.best_global_score:
            self.best_global_score = best_of_this_run
            self.best_global_matrix = best_matrix_of_run

            try:
                # Apenas um arquivo por execução, somente o melhor global validado
                results_dir = Path("memetic/results")
                results_dir.mkdir(exist_ok=True, parents=True)

                # Remove arquivos anteriores desta execução
                for f in results_dir.glob(f"*Run{run_id}*"):
                    f.unlink()

                matrix_file = results_dir / f"Run{run_id}-AdaptivePAM-{best_of_this_run:.4f}.txt"
                best_matrix_of_run.to_clustalw_format(matrix_file)
                self.logger.info(f"Run {run_id}: Saved validated best matrix (score: {best_of_this_run:.4f}) to: {matrix_file}")
            except Exception as e:
                self.logger.error(f"Run {run_id}: Error saving matrix: {e}")

        return best_matrix_of_run

    def _partial_restart(self):
        """Reinicia parte da população mantendo melhores soluções."""
        # Mantém top 20% dos indivíduos
        num_keep = max(1, int(self.population.size * 0.2))
        kept_individuals = sorted(
            self.population.individuals,
            key=lambda x: x.fitness,
            reverse=True
        )[:num_keep]

        # Reinicia resto da população
        new_individuals = []
        for _ in range(self.population.size - num_keep):
            if random.random() < 0.5:  # 50% chance de usar elite como template
                template = random.choice(kept_individuals)
                new_ind = self.population._mutate(
                    template,
                    mutation_rate=0.2  # Taxa maior para aumentar diversidade
                )
            else:  # Cria novo indivíduo do zero
                new_ind = Individual()

            new_individuals.append(new_ind)

        # Atualiza população
        self.population.individuals = kept_individuals + new_individuals
        self.logger.info(f"Partial restart: kept {num_keep} individuals")
