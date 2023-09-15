import asdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("/Users/josh/src/M1M3_ML/data/myUdn3norm_156.mat")

x = data['x'][:, 0]
y = data['y'][:, 0]
z = data['Udn3norm'].T

fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(7,9))
for i, ax in enumerate(axes.flat):
    ax.scatter(x, y, c=z[i], s=1, cmap='Spectral_r')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.savefig("M1M3_ML_modes.png")
# plt.show()


for fn, outfn in [
    ("/Users/josh/src/batoid_rubin/scripts/corrected/M1M3_corrected.asdf",
     "corrected_modes.png"),
    ("/Users/josh/src/batoid_rubin/scripts/improved/M1M3_improved.asdf",
     "improved_modes.png")
]:
    with asdf.open(fn) as af:
        x = np.array(af['fea_nodes']['X_Position'])
        y = np.array(af['fea_nodes']['Y_Position'])
        z = np.array(af['bend_1um']['sag'])

    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(7,9))
    for i, ax in enumerate(axes.flat):
        ax.scatter(x, y, c=z[i], s=1, cmap='Spectral_r')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.savefig(outfn)
    # plt.show()
