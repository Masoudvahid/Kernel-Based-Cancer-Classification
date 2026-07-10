import json

notebook_path = "radial_single_kernel_fi_optimization_tissue_mask.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells'][:14]):
    cell_type = cell['cell_type']
    source = cell.get('source', [])
    first_lines = "".join(source[:2])
    print(f"Cell {idx} ({cell_type}): {repr(first_lines[:120])}...")
    if cell_type == 'code' and 'outputs' in cell and cell['outputs']:
        print(f"  Has outputs. Number of outputs: {len(cell['outputs'])}")
        # print first few output texts
        for output in cell['outputs']:
            if output['output_type'] == 'stream':
                print("    Stream output: " + "".join(output.get('text', []))[:200])
