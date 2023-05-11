import numpy as np
from sklearn.datasets import fetch_openml
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


# Prikaz prvih nekoliko slika
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.set_title(y[i])
    ax.axis('off')

plt.show()


# skaliraj podatke, train/test split
X = X / 255.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# Kreiranje i treniranje modela
mlp_mnist = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, alpha=1e-4,
                          solver='sgd', verbose=10, random_state=1,
                          learning_rate_init=0.001)

mlp_mnist.fit(X_train, y_train)


# Izračun točnosti
train_accuracy = mlp_mnist.score(X_train, y_train)
test_accuracy = mlp_mnist.score(X_test, y_test)

print("Točnost na skupu podataka za učenje:", train_accuracy)
print("Točnost na skupu podataka za testiranje:", test_accuracy)


# Matrica zabune
y_train_pred = mlp_mnist.predict(X_train)
y_test_pred = mlp_mnist.predict(X_test)

train_confusion = confusion_matrix(y_train, y_train_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

print("Matrica zabune na skupu podataka za učenje:")
print(train_confusion)

print("Matrica zabune na skupu podataka za testiranje:")
print(test_confusion)


# Spremi mrežu na disk
filename = "NN_model.sav"
joblib.dump(mlp_mnist, filename)
