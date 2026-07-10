import json

notebook_path = "radial_single_kernel_fi_optimization_tissue_mask.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(14, len(nb['cells'])):
    cell = nb['cells'][idx]
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        print(f"\n--- Cell {idx} Outputs ---")
        for output in cell['outputs']:
            if output['output_type'] == 'stream':
                print("".join(output.get('text', [])))
            elif output['output_type'] == 'execute_result':
                print(output.get('data', {}).get('text/plain', ''))
