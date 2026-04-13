#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar imágenes PNG de los diagramas Mermaid.

Usa la API de mermaid.ink para renderizar los diagramas.

Uso:
    python generate_images.py

No requiere dependencias externas (usa urllib de la biblioteca estándar).
"""

import base64
import os
from pathlib import Path
import urllib.request
import urllib.error


def mermaid_to_png(mermaid_code: str, output_path: str) -> bool:
    """
    Convierte código Mermaid a PNG usando mermaid.ink API.

    Parameters
    ----------
    mermaid_code : str
        Código Mermaid del diagrama
    output_path : str
        Ruta donde guardar la imagen PNG

    Returns
    -------
    bool
        True si la conversión fue exitosa
    """
    # Codificar el diagrama en base64
    encoded = base64.urlsafe_b64encode(
        mermaid_code.encode('utf-8')
    ).decode('ascii')

    # URL de la API de mermaid.ink
    url = f"https://mermaid.ink/img/{encoded}"

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except urllib.error.HTTPError as e:
        print(f"Error HTTP: {e.code}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def mermaid_to_svg(mermaid_code: str, output_path: str) -> bool:
    """
    Convierte código Mermaid a SVG usando mermaid.ink API.

    Parameters
    ----------
    mermaid_code : str
        Código Mermaid del diagrama
    output_path : str
        Ruta donde guardar la imagen SVG

    Returns
    -------
    bool
        True si la conversión fue exitosa
    """
    # Codificar el diagrama en base64
    encoded = base64.urlsafe_b64encode(
        mermaid_code.encode('utf-8')
    ).decode('ascii')

    # URL de la API de mermaid.ink para SVG
    url = f"https://mermaid.ink/svg/{encoded}"

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except urllib.error.HTTPError as e:
        print(f"Error HTTP: {e.code}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Genera imágenes PNG y SVG de todos los diagramas .mmd."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "images"
    output_dir.mkdir(exist_ok=True)

    # Encontrar todos los archivos .mmd
    mmd_files = sorted(script_dir.glob("*.mmd"))

    if not mmd_files:
        print("No se encontraron archivos .mmd")
        return

    print(f"Encontrados {len(mmd_files)} diagramas Mermaid")
    print("=" * 50)

    for mmd_file in mmd_files:
        print(f"\nProcesando: {mmd_file.name}")

        # Leer el contenido
        with open(mmd_file, 'r', encoding='utf-8') as f:
            mermaid_code = f.read()

        # Generar PNG
        png_path = output_dir / f"{mmd_file.stem}.png"
        if mermaid_to_png(mermaid_code, str(png_path)):
            print(f"  ✓ PNG: {png_path.name}")
        else:
            print(f"  ✗ PNG: Error al generar")

        # Generar SVG
        svg_path = output_dir / f"{mmd_file.stem}.svg"
        if mermaid_to_svg(mermaid_code, str(svg_path)):
            print(f"  ✓ SVG: {svg_path.name}")
        else:
            print(f"  ✗ SVG: Error al generar")

    print("\n" + "=" * 50)
    print(f"Imágenes guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
