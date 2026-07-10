import json

notebook_path = "radial_single_kernel_fi_optimization_tissue_mask.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

output_lines = []
for idx in range(14, len(nb['cells'])):
    cell = nb['cells'][idx]
    output_lines.append(f"\n=================== CELL {idx} ({cell['cell_type']}) ===================")
    output_lines.append("".join(cell.get('source', [])))

with open("scratch_cells.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Cells written to scratch_cells.txt")
