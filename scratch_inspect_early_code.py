import json

notebook_path = "radial_single_kernel_fi_optimization_tissue_mask.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

output_lines = []
for idx in range(6, 14):
    cell = nb['cells'][idx]
    output_lines.append(f"\n=================== CELL {idx} ({cell['cell_type']}) ===================")
    output_lines.append("".join(cell.get('source', [])))

with open("scratch_early_cells.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Early cells written to scratch_early_cells.txt")
