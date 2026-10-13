"""Plot a recorded learned profile; no models are trained or results altered."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/exp_022_mmr_nperFamily500_nComposition1250/radial_kernel_bank_k16_r20/radial_kernel_bank_summary.json'
OUTPUT = ROOT / 'publication/figures/learned_profile_interpretation.png'
data = json.loads(SOURCE.read_text())
profile = np.asarray(data['learned_profiles'][13])  # one-based profile 14
size = data['config']['kernel_size'][0]
y, x = np.indices((size, size))
radius = np.floor(np.hypot(x - (size - 1) / 2, y - (size - 1) / 2)).astype(int)
kernel = profile[radius]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax, axk) = plt.subplots(2, 1, figsize=(4.3, 4.9), layout='constrained', gridspec_kw={'height_ratios': [1, 1.25]})
ax.axhline(0, color='0.55', linewidth=.7)
ax.plot(np.arange(len(profile)), profile, color='#204c7c', marker='o', markersize=3, linewidth=1.5)
ax.axvline(20.5, color='0.45', linestyle='--', linewidth=.8)
ax.set(xlabel='Radial bin r (pixels)', ylabel='Kernel weight', xlim=(-.5, 21.8), title='(a) Learned radial profile 14')
ax.set_xticks([0, 5, 10, 15, 20])
v = np.abs(kernel).max()
im = axk.imshow(kernel, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower', extent=(-15.5, 15.5, -15.5, 15.5), interpolation='nearest')
axk.set(xlabel='Horizontal offset (pixels)', ylabel='Vertical offset (pixels)', title='(b) Corresponding convolution kernel')
fig.colorbar(im, ax=axk, label='Kernel weight', shrink=.9, pad=.04)
fig.savefig(OUTPUT, dpi=240)
print(f'Wrote {OUTPUT}')
print('Profile 14 weights:', [data['metrics']['calib_w'][13 + offset] for offset in [0, 16, 32]])
print('Kernel weight sum:', float(kernel.sum()))
