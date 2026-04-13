#!/bin/bash

INPUT="$1"
MAX_SIZE=$((10 * 1024 * 1024)) # 10MB

# Obtener número de páginas con qpdf
PAGES=$(qpdf --show-npages "$INPUT")

# Extraer nombre base (sin extensión)
BASENAME=$(basename "$INPUT" .pdf)

start=1
part=1

while [ "$start" -le "$PAGES" ]; do
    end=$start

    while true; do
        OUT="${BASENAME}_part_${part}.pdf"
        qpdf "$INPUT" --pages "$INPUT" $start-$end -- "$OUT" >/dev/null 2>&1

        size=$(stat --printf="%s" "$OUT")

        # Si supera el tamaño máximo
        if [ "$size" -gt "$MAX_SIZE" ]; then
            rm -f "$OUT"
            end=$((end - 1))
            break
        fi

        # Si llegamos al final del PDF
        if [ "$end" -ge "$PAGES" ]; then
            break
        fi

        end=$((end + 1))
    done

    # Generar la parte final correcta
    OUT="${BASENAME}_part_${part}.pdf"
    qpdf "$INPUT" --pages "$INPUT" $start-$end -- "$OUT"

    echo "Generado ${OUT} (páginas $start-$end)"

    start=$((end + 1))
    part=$((part + 1))
done

