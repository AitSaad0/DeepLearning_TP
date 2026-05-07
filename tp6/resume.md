# TP 6 — Deep Learning
## Classification d’images avec un CNN sur Fashion MNIST

---

# 1. Objectif du TP

L’objectif de ce TP est de construire et entraîner un réseau de neurones convolutifs (CNN) afin de classifier les images du dataset Fashion MNIST.

Le modèle doit reconnaître différentes catégories de vêtements comme :
- chaussures,
- sacs,
- pulls,
- pantalons,
- chemises,
- etc.

---

# 2. Dataset utilisé : Fashion MNIST

Le dataset Fashion MNIST contient :

- 60 000 images d’entraînement
- 10 000 images de test
- images en niveaux de gris
- taille : 28 × 28 pixels
- 10 classes différentes

Chargement du dataset :

```python
from keras.datasets import fashion_mnist

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()