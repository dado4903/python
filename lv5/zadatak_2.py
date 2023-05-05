import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np


def generate_data(n_samples, flagc):

    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                          centers=4,
                          cluster_std=[1.0, 2.5, 0.5, 3.0],
                          random_state=random_state)

    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)

    else:
        X = []

    return X


# Generiranje 500 podataka
X = generate_data(500, 1)

# K-means s 1 do 20 klastera i spremanje vrijednosti kriterijske funkcije
scores = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    scores.append(kmeans.inertia_)

# Grafički prikaz vrijednosti kriterijske funkcije za broj klastera od 1 do 20
plt.plot(range(1, 21), scores)
plt.xlabel('Broj klastera')
plt.ylabel('Vrijednost kriterijske funkcije')
plt.show()
