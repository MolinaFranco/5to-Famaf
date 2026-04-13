#!/bin/bash

INPUT="$1"
MAX_SIZE=$((32 * 1024 * 1024)) # 32MB

# Obtener número de páginas con qpdf (sin pdfinfo)
PAGES=$(qpdf --show-npages "$INPUT")

start=1
part=1

while [ "$start" -le "$PAGES" ]; do
    end=$start

    while true; do
        OUT="part_$part.pdf"
        qpdf "$INPUT" --pages "$INPUT" $start-$end -- "$OUT" >/dev/null 2>&1

        size=$(stat --printf="%s" "$OUT")

        # Si el archivo generado supera el tamaño máximo
        if [ "$size" -gt "$MAX_SIZE" ]; then
            rm -f "$OUT"
            end=$((end - 1))
            break
        fi

        # Si ya llegamos al final del PDF
        if [ "$end" -ge "$PAGES" ]; then
            break
        fi

        end=$((end + 1))
    done

    # Generar la parte final buena
    qpdf "$INPUT" --pages "$INPUT" $start-$end -- "part_$part.pdf"
    echo "Generado part_$part.pdf (páginas $start-$end)"

    start=$((end + 1))
    part=$((part + 1))
done

