#!/usr/bin/env python3
"""Convert a notebook with SVG image tags into a notebook referencing PNG images"""
import os
import sys
import json
from bs4 import BeautifulSoup


def main():
    if len(sys.argv)<2 or "--help" in sys.argv:
        print("Usage: convert_nb.py old_filename.ipynb [new_filename.ipynb]")
        sys.exit(1)
    else:
        old_filename = sys.argv[1]
        if len(sys.argv) >= 3:
            new_filename = sys.argv[2]
        else:
            old_name, ext = os.path.splitext(old_filename)
            new_filename = old_name + '_png' + ext
        convert_notebook(old_filename, new_filename)


def convert_notebook(old_filename, new_filename):
    """Convert SVG references to PNG references in a Jupyter Notebook file.

    Looks for these: <img src="beautiful chart.svg" alt="Graphic">
    Replaces with these: ![Graphic](beautiful%20chart.png)

    Arguments
    ---------
    old_filename : str
    new_filename : str

    Returns
    -------
    None
    """
    with open(old_filename) as f:
        nb = json.load(f)
    md_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'markdown']
    for cell in md_cells:
        lines = cell.get('source', [])
        for i, line in enumerate(lines):
            if '<img' in line.lower():
                # image = BeautifulSoup(line, 'html.parser').select_one('img')
                # src = image.attrs.get('src', '').replace(' ', '%20').replace('.svg', '.png')
                # alt = image.attrs.get('alt', '')
                # new_line = '![{alt}]({src})'.format(alt=alt, src=src)
                lines[i] = line.replace('.svg', '.png')
    with open(new_filename, 'w') as f:
        json.dump(nb, f)


if __name__ == "__main__":
    main()
