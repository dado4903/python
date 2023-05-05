# -*- coding: utf-8 -*-
"""
Created on Sun Dec 02 12:08:00 2018

@author: Grbic
"""

import matplotlib.image as mpimg
import scipy as sp
from sklearn import cluster, datasets
import numpy as np
import matplotlib.pyplot as plt

try:
    face = sp.face(gray=True)
except AttributeError:
    from scipy import misc
    face = misc.face(gray=True)

X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

plt.figure(1)
plt.imshow(face,  cmap='gray')
plt.show()

plt.figure(2)
plt.imshow(face_compressed,  cmap='gray')
plt.show()

image = mpimg.imread('example_grayscale.png')
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
X = image.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
image_compressed = np.choose(labels, values)
image_compressed.shape = image.shape
plt.imshow(image_compressed, cmap='gray')
plt.axis('off')
plt.show()
compressed_size = image_compressed.size * image_compressed.itemsize
original_size = image.size * image.itemsize
compression_ratio = original_size / compressed_size
print("Compression ratio with 10 clusters: ", compression_ratio)
