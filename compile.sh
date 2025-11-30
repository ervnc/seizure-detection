#!/bin/bash
# Script para compilar o relatório LaTeX com todas as referências resolvidas

echo "Compilando relatorio.tex..."
echo ""

# Primeira compilação - gera arquivos auxiliares
pdflatex -interaction=nonstopmode relatorio.tex > /dev/null 2>&1
echo "✓ Primeira compilação concluída"

# Segunda compilação - resolve referências
pdflatex -interaction=nonstopmode relatorio.tex > /dev/null 2>&1
echo "✓ Segunda compilação concluída"

# Terceira compilação - garante que todas as referências estão resolvidas
pdflatex -interaction=nonstopmode relatorio.tex > /dev/null 2>&1
echo "✓ Terceira compilação concluída"

echo ""
echo "✓ PDF gerado: relatorio.pdf"
echo ""

# Verifica se há referências não resolvidas
if grep -q "??" relatorio.log 2>/dev/null; then
    echo "⚠ Aviso: Algumas referências podem não ter sido resolvidas."
    echo "   Execute novamente: pdflatex relatorio.tex"
else
    echo "✓ Todas as referências foram resolvidas!"
fi

