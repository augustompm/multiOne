#!/bin/bash

# Nome do arquivo oculto para armazenar o número de referência
REF_FILE=".ref_number"

# Função para exibir mensagens de erro e sair
function error_exit {
    echo "$1" 1>&2
    exit 1
}

# Função para exibir avisos
function warning {
    echo "WARNING: $1" 1>&2
}

# Verifica se o arquivo de referência existe. Se não, cria com 001
if [ ! -f "$REF_FILE" ]; then
    echo "001" > "$REF_FILE" || error_exit "Erro ao criar o arquivo de referência."
fi

# Lê o número de referência atual
CURRENT_REF=$(cat "$REF_FILE")

# Verifica se o número de referência está no formato correto (três dígitos)
if ! [[ "$CURRENT_REF" =~ ^[0-9]{3}$ ]]; then
    echo "Formato inválido no $REF_FILE. Reiniciando para 001."
    CURRENT_REF="001"
    echo "$CURRENT_REF" > "$REF_FILE" || error_exit "Erro ao reiniciar o número de referência."
fi

# Executa git add
git add . || error_exit "Erro ao executar 'git add .'."

# Executa git commit com a mensagem "V0.<número>"
git commit -m "V0.${CURRENT_REF}" || {
    echo "Nenhuma alteração para commitar."
    exit 0
}

# Executa git push para a branch main
git push origin main || error_exit "Erro ao executar 'git push origin main'."

# Incrementa o número de referência
NEXT_REF=$(printf "%03d" $((10#$CURRENT_REF + 1)))

# Verifica se o número excedeu 999 e reinicia se necessário
if [ "$NEXT_REF" -gt "999" ]; then
    echo "O número de referência excedeu 999. Reiniciando para 001."
    NEXT_REF="001"
fi

# Atualiza o arquivo de referência com o próximo número
echo "$NEXT_REF" > "$REF_FILE" || error_exit "Erro ao atualizar o número de referência."

echo "Commit realizado com sucesso com a referência V0.${CURRENT_REF}."
echo "Próximo número de referência: V0.${NEXT_REF}."
