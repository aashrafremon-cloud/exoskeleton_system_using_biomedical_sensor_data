import json
import sys

def extract_code(filename, out_f):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        out_f.write(f'--- {filename} ---\n')
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                lines = cell.get('source', [])
                if not lines:
                    continue
                out_f.write(''.join(lines))
                out_f.write('\n# --- end of cell ---\n')
    except Exception as e:
        out_f.write(f"Error {filename}: {e}\n")

with open('extracted_code_utf8.py', 'w', encoding='utf-8') as out_f:
    extract_code('EDA&Preprocessing (1).ipynb', out_f)
    extract_code('Preprocessing_georgia.ipynb', out_f)
