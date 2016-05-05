__author__ = 'Charlie'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

ap = argparse.ArgumentParser("Script to show checkpoint plots")
ap.add_argument("-f", "--file", required=True, help="Path to checkpoint file")
args = vars(ap.parse_args())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
colors = np.random.rand(10)
arr = np.genfromtxt(args["file"], dtype=float, delimiter=" ")
indices = np
ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=10, c=colors[arr[:,3].astype(np.int)], marker='o')
plt.show()