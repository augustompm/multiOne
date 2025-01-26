#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: biofit.py
Descrição: Calcula o WSP (Weighted Sum of Pairs) normalizado para alinhamentos
múltiplos de sequências proteicas.

Fundamentação Teórica:

1. WSP (Weighted Sum-of-Pairs) Score
O WSP é uma medida da qualidade de um alinhamento múltiplo que leva em conta
as relações filogenéticas entre as sequências. Conforme Gotoh (1995):

WSP(A) = Σ Σ wjk·Sjk
         j k<j

Onde:
- wjk são os pesos para cada par de sequências j,k
- Sjk é o score entre as sequências j e k no alinhamento A

2. Pesos Filogenéticos
Os pesos wjk são derivados de uma árvore filogenética UPGMA para compensar
por representação desigual de subgrupos (Gotoh 1994). O peso de um par é
calculado usando as distâncias na árvore:

wjk = djk / Σ dij
      i,j

Onde:
- djk é a distância entre as sequências j e k na árvore
- A soma no denominador é sobre todas as distâncias

3. Normalização do WSP
O WSP é normalizado para o intervalo [0,1] usando:

WSP_norm = (WSP - WSP_min) / (WSP_max - WSP_min)

Onde WSP_min e WSP_max são os scores mínimo e máximo possíveis
dados os scores na matriz de substituição.

4. Matrizes de Substituição
As matrizes AdaptivePAM são lidas de arquivos .txt e convertidas em
scores de substituição para o cálculo do WSP. A normalização preserva
as proporções relativas entre os scores.

Referências:
- Gotoh O (1994) JMB 264:823-838
- Gotoh O (1995) CABIOS 11:543-551
- Thompson et al. (1999) NAR 27:2682-2690

Uso:
O script espera:
1. Arquivos FASTA com alinhamentos em ./clustalw/
2. Matrizes AdaptivePAM em ./AdaptivePAM/
3. Gera resultados em wsp_scores_normalized.csv
"""

import os
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Phylo.BaseTree import Tree, Clade
from Bio.Align import MultipleSeqAlignment


class SubstitutionMatrix:
    """
    Classe para manipulação de matrizes de substituição de aminoácidos.
    
    Implementa leitura de matrizes AdaptivePAM e cálculo de scores
    de substituição para uso no WSP.
    
    Atributos:
        name: Nome da matriz
        scores: Dicionário com scores de substituição
        amino_acids: Conjunto de aminoácidos na matriz
    """
    
    def __init__(self, name: str = ""):
        """
        Inicializa matriz vazia.
        
        Args:
            name: Nome da matriz
        """
        self.name = name
        self.scores = {}  # Dict[(aa1,aa2)] -> score
        self.amino_acids = set()
        
    @classmethod
    def from_file(cls, file_path: Path) -> 'SubstitutionMatrix':
        """
        Cria matriz a partir de arquivo AdaptivePAM.
        
        O formato esperado é:
        - Primeira linha: aminoácidos das colunas
        - Linhas seguintes: aminoácido da linha seguido dos scores
        - '#' indica linha de comentário
        
        Args:
            file_path: Caminho para arquivo .txt
            
        Returns:
            Nova instância de SubstitutionMatrix
        
        Raises:
            Exception: Se houver erro na leitura/parse do arquivo
        """
        matrix = cls(name=file_path.stem)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Remove comentários e linhas vazias
            lines = [line.strip() for line in lines 
                    if not line.startswith('#') and line.strip()]
            
            # Primeira linha tem aminoácidos
            header = lines[0].split()
            
            # Parse scores linha a linha
            for line in lines[1:]:
                parts = line.split()
                row_aa = parts[0]
                scores = parts[1:]
                
                for col_aa, score in zip(header, scores):
                    try:
                        score_value = float(score)
                    except ValueError:
                        print(f"Aviso: Score inválido '{score}'. Usando 0.0")
                        score_value = 0.0
                        
                    # Armazena aminoácidos únicos
                    matrix.amino_acids.add(row_aa)
                    matrix.amino_acids.add(col_aa)
                    
                    # Armazena score de forma ordenada (aa1 <= aa2)
                    key = tuple(sorted([row_aa, col_aa]))
                    matrix.scores[key] = score_value
                    
            return matrix
            
        except Exception as e:
            print(f"Erro ao ler matriz AdaptivePAM: {e}")
            raise
            
    def get_score(self, aa1: str, aa2: str) -> float:
        """
        Retorna score de substituição para par de aminoácidos.
        
        Args:
            aa1, aa2: Aminoácidos (caso-insensitivo)
            
        Returns:
            Score de substituição normalizado
        """
        key = tuple(sorted([aa1.upper(), aa2.upper()]))
        return self.scores.get(key, 0.0)
        
    def normalize_scores(self) -> None:
        """
        Normaliza scores para o intervalo [0,1].
        
        Mantém proporções relativas entre os scores.
        """
        if not self.scores:
            return
            
        max_score = max(self.scores.values())
        min_score = min(self.scores.values())
        range_score = max_score - min_score
        
        if range_score > 0:
            for key in self.scores:
                self.scores[key] = (self.scores[key] - min_score) / range_score


class BiopythonMatrixWrapper:
    """
    Wrapper para matrizes de substituição do Biopython.
    Adapta a interface para ser compatível com nossa SubstitutionMatrix.
    """
    def __init__(self, matrix, name: str):
        """
        Args:
            matrix: Matriz do Biopython (Array)
            name: Nome da matriz
        """
        self.matrix = matrix
        self.name = name
        self.alphabet = matrix.alphabet
        
        # Adiciona o atributo scores necessário para calculate_wsp
        self.scores = {}
        for i, aa1 in enumerate(self.alphabet):
            for j, aa2 in enumerate(self.alphabet):
                key = tuple(sorted([aa1, aa2]))
                self.scores[key] = float(self.matrix[i][j])
        
    def get_score(self, aa1: str, aa2: str) -> float:
        """
        Retorna score de substituição usando indexação do Biopython.
        
        Args:
            aa1, aa2: Aminoácidos
            
        Returns:
            Score de substituição
        """
        try:
            i = self.alphabet.index(aa1.upper())
            j = self.alphabet.index(aa2.upper())
            return float(self.matrix[i][j])
        except (ValueError, IndexError):
            return 0.0


def calculate_sequence_distances(alignment: MultipleSeqAlignment,
                              matrix: SubstitutionMatrix) -> Dict[str, Dict[str, float]]:
    """
    Calcula matriz de distâncias entre todas as sequências do alinhamento.
    
    As distâncias são baseadas nos scores de substituição normalizados:
    dist(s1,s2) = 1 - média(scores de substituição)
    
    Args:
        alignment: Alinhamento múltiplo
        matrix: Matriz de substituição
        
    Returns:
        Matriz de distâncias nome1->nome2->distância
    """
    distances = defaultdict(dict)
    
    for i, seq1 in enumerate(alignment):
        for j, seq2 in enumerate(alignment):
            if i < j:  # Calcula apenas metade superior
                score = 0
                valid_positions = 0
                
                # Soma scores para posições não-gap
                for aa1, aa2 in zip(str(seq1.seq), str(seq2.seq)):
                    if aa1 != '-' and aa2 != '-':
                        score += matrix.get_score(aa1, aa2)
                        valid_positions += 1
                
                # Calcula distância como 1 - score_médio
                if valid_positions > 0:
                    avg_score = score / valid_positions
                    distance = 1 - avg_score
                else:
                    distance = 1.0
                    
                # Armazena distância simétrica
                distances[seq1.id][seq2.id] = distance
                distances[seq2.id][seq1.id] = distance
                
        # Distância para própria sequência = 0
        distances[seq1.id][seq1.id] = 0.0
        
    return distances


def generate_upgma_tree(alignment: MultipleSeqAlignment,
                       matrix: SubstitutionMatrix) -> Optional[Tree]:
    """
    Gera árvore UPGMA usando distâncias calculadas da matriz.
    
    O método UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
    é usado por ser um método hierárquico que produz uma árvore 
    ultramétrica adequada para o cálculo de pesos filogenéticos.
    
    Args:
        alignment: Alinhamento múltiplo
        matrix: Matriz de substituição
        
    Returns:
        Tree UPGMA ou None se erro
    """
    try:
        # Calcula matriz de distâncias
        distances = calculate_sequence_distances(alignment, matrix)
        
        # Cria calculador e construtor de árvore
        calculator = DistanceCalculator()
        calculator._dmat = distances  # Define matriz de distâncias
        
        constructor = DistanceTreeConstructor(calculator, method="upgma")
        tree = constructor.build_tree(alignment)
        
        return tree
        
    except Exception as e:
        print(f"Erro ao gerar árvore UPGMA: {e}")
        return None


def get_tree_distances(tree: Tree, names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calcula distâncias entre pares de folhas na árvore.
    
    A distância é calculada somando os branch_lengths no caminho entre cada par
    de nós terminais.
    
    Args:
        tree: Árvore filogenética
        names: Lista de nomes das sequências
        
    Returns:
        Matriz de distâncias nome1->nome2->distância
    """
    distances = defaultdict(lambda: defaultdict(float))
    
    # Mapeia nomes para nós terminais
    terminals = {}
    for leaf in tree.get_terminals():
        terminals[leaf.name] = leaf
    
    # Calcula distâncias entre pares
    for name1 in names:
        for name2 in names:
            if name1 != name2:
                try:
                    leaf1 = terminals[name1]
                    leaf2 = terminals[name2]
                    
                    # Obtém caminho da raiz até cada nó
                    path1 = tree.get_path(leaf1)
                    path2 = tree.get_path(leaf2)
                    
                    # Encontra o ancestral comum mais próximo (LCA)
                    # comparando os caminhos da raiz
                    i = 0
                    while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
                        i += 1
                    
                    # Distância é a soma dos branch_lengths no caminho:
                    # - da folha 1 até o LCA
                    # - do LCA até a folha 2
                    dist = 0.0
                    for node in path1[i:]:
                        dist += node.branch_length or 0.0
                    for node in path2[i:]:
                        dist += node.branch_length or 0.0
                    
                    distances[name1][name2] = dist
                    distances[name2][name1] = dist
                    
                except KeyError:
                    print(f"Aviso: Nó não encontrado para {name1} ou {name2}")
                except Exception as e:
                    print(f"Erro ao calcular distância entre {name1} e {name2}: {e}")
                
    return distances


def normalize_weights(distances: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normaliza pesos para soma = 1.
    
    Args:
        distances: Matriz de distâncias
        
    Returns:
        Matriz de pesos normalizados
    """
    weights = defaultdict(lambda: defaultdict(float))
    total = sum(sum(d.values()) for d in distances.values())
    
    if total > 0:
        for name1 in distances:
            for name2 in distances[name1]:
                weights[name1][name2] = distances[name1][name2] / total
                
    return weights


def normalize_wsp(wsp: float, matrix: SubstitutionMatrix) -> float:
    """
    Normaliza WSP usando Z-Score seguido de MinMax, mantendo a natureza
    de distância das matrizes PAM (valores maiores = maior distância = pior alinhamento).
    
    Args:
        wsp: WSP score bruto
        matrix: Matriz de substituição
        
    Returns:
        WSP normalizado [0,1], onde valores maiores indicam maior distância
    """
    # Calcula estatísticas dos scores da matriz
    scores = list(matrix.scores.values())
    mean_score = sum(scores) / len(scores)
    variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
    std_dev = variance ** 0.5
    
    if std_dev == 0:
        return 0.0
        
    # Z-Score do WSP
    wsp_zscore = (wsp - mean_score) / std_dev
    
    # Z-Score de todos os valores para encontrar min/max
    min_zscore = (min(scores) - mean_score) / std_dev
    max_zscore = (max(scores) - mean_score) / std_dev
    
    # MinMax no Z-Score mantendo a natureza de distância
    # (maior valor = maior distância = pior alinhamento)
    if max_zscore > min_zscore:
        wsp_norm = (wsp_zscore - min_zscore) / (max_zscore - min_zscore)
    else:
        wsp_norm = 0.0
    
    return max(0.0, min(1.0, wsp_norm))

def calculate_wsp(alignment: MultipleSeqAlignment,
                 tree: Tree,
                 matrix: SubstitutionMatrix) -> tuple[float, float]:
    """
    Calcula WSP score bruto e normalizado.
    
    Returns:
        Tupla (wsp_bruto, wsp_normalizado)
    """
    try:
        sequences = [str(record.seq) for record in alignment]
        names = [record.id for record in alignment]
        
        distances = get_tree_distances(tree, names)
        weights = normalize_weights(distances)
        
        wsp = 0.0
        total_weight = 0.0
        
        # Calcula WSP bruto
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                weight = weights[names[i]][names[j]]
                seq1, seq2 = sequences[i], sequences[j]
                
                score = 0
                comparable_positions = 0
                
                for aa1, aa2 in zip(seq1, seq2):
                    if aa1 != '-' and aa2 != '-':
                        score += matrix.get_score(aa1, aa2)
                        comparable_positions += 1
                        
                if comparable_positions > 0:
                    avg_score = score / comparable_positions
                    wsp += weight * avg_score
                    total_weight += weight
                    
        if total_weight > 0:
            wsp_raw = wsp / total_weight
        else:
            wsp_raw = 0.0
            
        # Estatísticas da matriz para normalização
        scores = list(matrix.scores.values())
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return wsp_raw, 0.0
            
        # Z-Score
        wsp_zscore = (wsp_raw - mean_score) / std_dev
        min_zscore = (min(scores) - mean_score) / std_dev
        max_zscore = (max(scores) - mean_score) / std_dev
        
        # MinMax mantendo natureza de distância
        if max_zscore > min_zscore:
            wsp_norm = (wsp_zscore - min_zscore) / (max_zscore - min_zscore)
        else:
            wsp_norm = 0.0
            
        return wsp_raw, max(0.0, min(1.0, wsp_norm))
        
    except Exception as e:
        print(f"Erro ao calcular WSP: {e}")
        return 0.0, 0.0

def get_substitution_matrices(fasta_name: str, adaptive_pam_dir: Path) -> Dict[str, Any]:
    """
    Retorna apenas PAM250 e AdaptivePAM para comparação
    """
    matrices = {}
    
    # Carrega apenas PAM250 do Biopython
    try:
        from Bio.Align import substitution_matrices
        matrices["PAM250"] = BiopythonMatrixWrapper(
            substitution_matrices.load("PAM250"),
            "PAM250"
        )
    except Exception as e:
        print(f"Erro ao carregar PAM250: {e}")
        return {}
        
    # Carrega matriz AdaptivePAM correspondente
    adaptive_pam_file = adaptive_pam_dir / f"{fasta_name}.txt"
    if adaptive_pam_file.exists():
        try:
            matrices[fasta_name] = SubstitutionMatrix.from_file(adaptive_pam_file)
        except Exception as e:
            print(f"Erro ao carregar matriz AdaptivePAM {fasta_name}: {e}")
    else:
        print(f"Matriz AdaptivePAM não encontrada para {fasta_name}")
        
    return matrices

def main():
    """Função principal."""
    
    # Caminhos
    clustalw_dir = Path("clustalw/")
    adaptive_pam_dir = Path("AdaptivePAM/")
    output_csv = Path("wsp_scores_normalized.csv")
    
    # Verifica diretórios
    if not clustalw_dir.exists():
        print(f"Diretório {clustalw_dir} não existe")
        sys.exit(1)
    if not adaptive_pam_dir.exists():
        print(f"Diretório {adaptive_pam_dir} não existe")
        sys.exit(1)
                
    # Processa alinhamentos
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sequence", 
                        "PAM250_raw", "PAM250_norm",
                        "AdaptivePAM_raw", "AdaptivePAM_norm"])
        
        # Para cada arquivo FASTA
        fasta_files = list(clustalw_dir.glob("*.fasta"))
        print(f"\nProcessando {len(fasta_files)} alinhamentos")
        
        for fasta_file in fasta_files:
            name = fasta_file.stem
            print(f"\nProcessando alinhamento: {name}")
            
            try:
                # Lê alinhamento
                alignment = AlignIO.read(fasta_file, "fasta")
                print(f"  {len(alignment)} sequências, comprimento {alignment.get_alignment_length()}")
                
                # Carrega as três matrizes para este alinhamento
                matrices = get_substitution_matrices(name, adaptive_pam_dir)
                if not matrices:
                    print(f"  Pulando {name} - matrizes não disponíveis")
                    continue
                
                # Calcula WSP para cada matriz
                results = {"sequence": name}
                
                # Análise dos scores da matriz
                for matrix_name, matrix in matrices.items():
                    scores = list(matrix.scores.values())
                    print(f"\n  Estatísticas da matriz {matrix_name}:")
                    print(f"    Min score: {min(scores):.4f}")
                    print(f"    Max score: {max(scores):.4f}")
                    print(f"    Média: {sum(scores)/len(scores):.4f}")
                    var = sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores)
                    print(f"    Variância: {var:.4f}")
                
                for matrix_name, matrix in matrices.items():
                    print(f"\n  Calculando WSP com matriz {matrix_name}...")
                    try:
                        tree = generate_upgma_tree(alignment, matrix)
                        if tree is None:
                            results[matrix_name] = "NA"
                            continue
                            
                        wsp_raw, wsp_norm = calculate_wsp(alignment, tree, matrix)
                        print(f"    WSP Bruto: {wsp_raw:.4f}")
                        print(f"    WSP Normalizado: {wsp_norm:.4f}")
                        
                        if matrix_name == name:
                            results["AdaptivePAM_raw"] = f"{wsp_raw:.4f}"
                            results["AdaptivePAM_norm"] = f"{wsp_norm:.4f}"
                        else:
                            results["PAM250_raw"] = f"{wsp_raw:.4f}"
                            results["PAM250_norm"] = f"{wsp_norm:.4f}"
                        
                    except Exception as e:
                        print(f"  Erro processando matriz {matrix_name}: {e}")
                        if matrix_name == name:
                            results["AdaptivePAM_raw"] = "NA"
                            results["AdaptivePAM_norm"] = "NA"
                        else:
                            results["PAM250_raw"] = "NA"
                            results["PAM250_norm"] = "NA"
                
                writer.writerow([
                    results["sequence"],
                    results.get("PAM250_raw", "NA"),
                    results.get("PAM250_norm", "NA"),
                    results.get("AdaptivePAM_raw", "NA"),
                    results.get("AdaptivePAM_norm", "NA")
                ])
                    
            except Exception as e:
                print(f"Erro processando alinhamento {name}: {e}")
                writer.writerow([name, "NA", "NA", "NA", "NA"])
                continue
                
    print(f"\nProcessamento concluído. Resultados salvos em {output_csv}")


if __name__ == "__main__":
    main()