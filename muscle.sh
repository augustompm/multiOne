#!/bin/bash
#
# Exemplo de uso do MUSCLE 5.3 conforme o help disponível:
#   muscle -align input.fa -output alignment.afa
#
# Rodamos em cada arquivo .tfa dentro de BAliBASE/RV30 e salvamos no diretório /muscle_isolado.

# 1) Ajuste as variáveis abaixo se precisar
MUSCLE_BIN="/dados/home/tesla-dados/multione/muscle-5.3/src/muscle-linux"
CR3_DIR="/dados/home/tesla-dados/multione/BAliBASE/RV30"
OUTPUT_DIR="/dados/home/tesla-dados/multione/results/muscle_isolado"

# 2) Cria diretório de saída (caso não exista)
mkdir -p "$OUTPUT_DIR"

echo "Rodando MUSCLE 5.3 no diretório: $CR3_DIR"
echo "Resultados irão para: $OUTPUT_DIR"
echo

# 3) Loop em cada arquivo .tfa
for fasta in "$CR3_DIR"/*.tfa; do
    # extrai apenas o nome do arquivo sem extensão .tfa
    base=$(basename "$fasta" .tfa)
    # onde vamos salvar o alinhamento
    out="$OUTPUT_DIR/${base}.aln"

    echo "Alinhando $fasta -> $out"

    # 4) Usa a sintaxe do help: -align / -output
    #    Gera saída em formato FASTA
    "$MUSCLE_BIN" \
        -align  "$fasta" \
        -output "$out"

    # 5) Verifica se houve erro
    if [ $? -ne 0 ]; then
        echo "ERRO ao alinhar $fasta"
    else
        echo "OK: Alinhamento gerado em $out"
    fi

    echo
done

echo "Processo concluído."
