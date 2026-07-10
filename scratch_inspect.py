import json

notebook_path = "radial_single_kernel_fi_optimization_tissue_mask.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("\n--- Last 10 cells summary ---")
for idx, cell in enumerate(nb['cells'][-10:]):
    real_idx = len(nb['cells']) - 10 + idx
    cell_type = cell['cell_type']
    source = cell.get('source', [])
    first_lines = "".join(source[:3])
    print(f"Cell {real_idx} ({cell_type}): {repr(first_lines[:100])}...")
