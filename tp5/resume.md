# 🧠 TP – Réseau de Neurones Convolutif sur CIFAR-10

## 🎯 Objectif du TP
Implémenter et analyser un réseau de neurones convolutif (CNN) appliqué au dataset CIFAR-10 afin d’étudier l’impact de différentes techniques :
- normalisation
- data augmentation (crop)
- padding
- dropout

---

# 📦 1. Dataset CIFAR-10

Le dataset CIFAR-10 contient :
- 60 000 images couleur (32×32×3)
- 10 classes (avion, voiture, chat, chien, etc.)
- 50 000 images d’entraînement
- 10 000 images de test

---

# 🧱 2. Architecture du CNN

Le modèle utilisé est composé de :

- Conv2D (32 filtres 5×5) + ReLU
- MaxPooling 2×2
- Conv2D (64 filtres 5×5) + ReLU
- MaxPooling 2×2
- Conv2D (64 filtres 5×5) + ReLU
- MaxPooling 2×2
- Flatten
- Dense (1000 neurones + ReLU)
- Dense (10 neurones + Softmax)

---

# 📏 3. Normalisation

Les images sont normalisées avec :

\[
x' = \frac{x - \mu}{\sigma}
\]

où :

- μ = [0.491, 0.482, 0.447]
- σ = [0.202, 0.199, 0.201]

### 🎯 Impact
- convergence plus rapide
- gradients plus stables
- meilleure performance

---

# 🧪 4. Data Augmentation (Random Crop)

Technique utilisée :
- ZeroPadding
- RandomCrop 28×28
- Resize 32×32

### 🎯 Objectif
- augmenter la diversité des données
- améliorer la robustesse

### ⚠️ Limite
- perte d’information sur les images petites
- peut dégrader certaines performances

---

# 🧮 5. Padding dans la convolution

Formule :

\[
Output = \frac{N - F + 2P}{S} + 1
\]

### Cas :
- sans padding → réduction de taille
- avec padding → conservation de la taille

### Exemple :
- padding = 2 pour filtre 5×5
- permet de garder la dimension

---

# 🧯 6. Dropout

Le dropout désactive aléatoirement des neurones pendant l’entraînement.

\[
\tilde{x}_i = x_i \cdot m_i
\quad \text{où } m_i \sim Bernoulli(1-p)
\]

### 🎯 Effet :
- réduit l’overfitting
- améliore la généralisation
- empêche la mémorisation

---

# 📊 7. Résultats expérimentaux

| Configuration | Test Accuracy |
|--------------|--------------|
| CNN simple | ~70% |
| + Normalisation | meilleure stabilité |
| + Data augmentation | ~63–70% |
| + Dropout | amélioration de la généralisation |

---

# 📉 8. Analyse globale

### ❌ Problèmes observés :
- overfitting sans régularisation
- perte d’information avec crop agressif
- sensibilité aux variations des données

### ✅ Améliorations :
- normalisation indispensable
- dropout améliore la généralisation
- augmentation augmente la robustesse

---

# 🚀 Conclusion

Ce TP montre que la performance d’un CNN ne dépend pas seulement de l’architecture, mais aussi fortement de :

- la qualité des données
- la normalisation
- les techniques de régularisation
- la data augmentation

L’ensemble de ces techniques permet d’obtenir un modèle plus robuste et généralisable sur CIFAR-10.

### Test accuracy : 0.6480000019073486
