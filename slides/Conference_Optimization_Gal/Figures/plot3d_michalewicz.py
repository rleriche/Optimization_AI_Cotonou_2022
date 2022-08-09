from matplotlib import cm  # color map
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 3.2, 50)
Y = np.linspace(0, 3.2, 50)
X, Y = np.meshgrid(X, Y)

# Michalewicz function
Z = -1 * (
    (np.sin(X) * np.sin((1 * X**2) / np.pi) ** 20)
    + (np.sin(Y) * np.sin((2 * Y**2) / np.pi) ** 20)
)

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, edgecolor="darkred", linewidth=0.1
)

ax.set_xlabel("x", fontsize=10)
ax.set_ylabel("y", fontsize=10)
ax.set_zlabel("f(x,y)", fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=6)

plt.show()
