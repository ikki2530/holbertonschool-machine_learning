#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

"""
- data: is a np.ndarray of shape (150, 4)
    150: number of flowers
    4: petal length, petal width, sepal length, sepal width
- labels: is a np.ndarray of shape (150,) containing information
about what species of iris each data point represents:
    0 -> Iris Setosa
    1 -> Iris Versicolor
    2 -> Iris Virginica

- pca_data: is a np.ndarray of shape (150, 3)
    The columns of pca_data represent the 3 dimensions of the reduced data,
    i.e., x, y, and z, respectively

"""
lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# your code here
fig = plt.figure()
ax = Axes3D(fig)
# using plasma color map, and labels to classify the data by colors
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels,
           cmap="plasma")
ax.set_title("PCA of Iris Dataset")
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.show()
