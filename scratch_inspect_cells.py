import json

notebook_path = "radial_single_kernel_fi_optimization_tissue_mask.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(14, 21):
    if idx < len(nb['cells']):
        cell = nb['cells'][idx]
        print(f"\n=================== CELL {idx} ({cell['cell_type']}) ===================")
        print("".join(cell.get('source', [])))
