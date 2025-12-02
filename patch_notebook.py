import nbformat

nb_path = '/Users/gracjanziemianski/Documents/CPC-SNN-GravitationalWavesDetection/notebooks/06_Phase1_Training_Report.ipynb'

with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'plot_case_study(model, dataset, dataset.index_list.index(idx_pure_noise)' in cell.source:
            print("Found cell to patch")
            cell.source = cell.source.replace('dataset.index_list.index(idx_pure_noise)', 'dataset.ids.index(str(idx_pure_noise))')
            cell.source = cell.source.replace('dataset.index_list.index(idx_glitch)', 'dataset.ids.index(str(idx_glitch))')
            cell.source = cell.source.replace('dataset.index_list.index(idx_gw)', 'dataset.ids.index(str(idx_gw))')

with open(nb_path, 'w') as f:
    nbformat.write(nb, f)

print("Notebook patched successfully")
