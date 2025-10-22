import json

with open(r"10.16.25 - Reps_Margins\conjunctive_lazy_rich_1e-5_reps_margins_v2\conjunctive_lazy_rich_1e_5_reps_margins.ipynb", 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, cell in enumerate(data['cells'][:50]):
    if cell.get('source'):
        first_line = cell['source'][0] if isinstance(cell['source'], list) else str(cell['source'])[:50]
        if 'sgd = 0.' in first_line or 'positions = [0.00' in first_line:
            print(f"\n\n===== Cell {i} =====")
            if isinstance(cell['source'], list):
                print(''.join(cell['source'][:20]))
            else:
                print(cell['source'][:500])

