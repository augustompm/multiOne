import os
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio import Phylo

def create_neighbor_joining_trees():
    # Lista das instâncias específicas do BAliBASE
    INSTANCES = [
        'BBA0004', 'BBA0005', 'BBA0008', 'BBA0011', 'BBA0014',
        'BBA0015', 'BBA0019', 'BBA0021', 'BBA0022', 'BBA0024',
        'BBA0080', 'BBA0126', 'BBA0133', 'BBA0142', 'BBA0148',
        'BBA0155', 'BBA0163', 'BBA0178', 'BBA0183', 'BBA0185',
        'BBA0192', 'BBA0201', 'BBA0218'
    ]
    
    # Caminhos
    balibase_path = "BAliBASE/RV100"          # Diretório contendo os arquivos .tfa
    output_path = "biofit/trees"              # Diretório para salvar as árvores
    
    # Criar diretório de saída se não existir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for instance in INSTANCES:
        input_file = os.path.join(balibase_path, f"{instance}.tfa")
        tree_file = os.path.join(output_path, f"{instance}_nj.newick")
        
        if not os.path.exists(input_file):
            print(f"Arquivo de alinhamento não encontrado: {input_file}")
            continue
        
        try:
            # Ler o alinhamento no formato FASTA
            alignment = AlignIO.read(input_file, "fasta")
            
            # Calcular a matriz de distância utilizando o modelo de identidade
            calculator = DistanceCalculator('identity')  # Outros modelos disponíveis: 'blosum62', 'transformation', etc.
            distance_matrix = calculator.get_distance(alignment)
            
            # Construir a árvore Neighbor-Joining
            constructor = DistanceTreeConstructor()
            nj_tree = constructor.nj(distance_matrix)
            
            # Salvar a árvore no formato Newick
            Phylo.write(nj_tree, tree_file, "newick")
            print(f"Árvore Neighbor-Joining gerada com sucesso para {instance}")
        
        except Exception as e:
            print(f"Erro ao gerar árvore para {instance}: {str(e)}")

if __name__ == "__main__":
    create_neighbor_joining_trees()
