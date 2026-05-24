import json
import sys

def extract_code(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        print(f'--- {filename} ---')
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                lines = cell.get('source', [])
                if not lines:
                    continue
                print(''.join(lines))
                print('\n# --- end of cell ---\n')
    except Exception as e:
        print(f"Error {filename}: {e}")

extract_code('EDA&Preprocessing (1).ipynb')
extract_code('Preprocessing_georgia.ipynb')
