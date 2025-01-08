#!/bin/bash

# Shell script para limpar os diretórios de resultados e o arquivo de log

# Saia imediatamente se um comando falhar
set -e

# Definição dos diretórios e arquivo de log
CLUSTALW_DIR="/dados/home/tesla-dados/multione/results/clustalw"
MUSCLE_DIR="/dados/home/tesla-dados/multione/results/muscle"
LOG_FILE="/dados/home/tesla-dados/multione/alignment_pipeline.log"

# Função para deletar conteúdos de um diretório
delete_directory_contents() {
    local DIR="$1"
    if [ -d "$DIR" ]; then
        echo "Deletando conteúdos de $DIR..."
        # Ativa a opção para incluir arquivos ocultos
        shopt -s dotglob
        # Remove todos os arquivos e subdiretórios
        rm -rf "$DIR"/*
        # Desativa a opção para evitar efeitos colaterais
        shopt -u dotglob
        echo "Conteúdos de $DIR deletados com sucesso."
    else
        echo "Diretório $DIR não existe. Pulando..."
    fi
}

# Função para deletar o arquivo de log
delete_log_file() {
    local LOG="$1"
    if [ -f "$LOG" ]; then
        echo "Deletando arquivo de log $LOG..."
        rm -f "$LOG"
        echo "Arquivo de log $LOG deletado com sucesso."
    else
        echo "Arquivo de log $LOG não existe. Pulando..."
    fi
}

# Deleta conteúdos dos diretórios ClustalW e MUSCLE
delete_directory_contents "$CLUSTALW_DIR"
delete_directory_contents "$MUSCLE_DIR"

# Deleta o arquivo de log
delete_log_file "$LOG_FILE"

echo "Processo de limpeza concluído."

exit 0
