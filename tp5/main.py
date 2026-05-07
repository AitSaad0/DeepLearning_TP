import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Chargement CIFAR-10
# =========================

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalisation SIMPLE seulement [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
mean = np.array([0.491, 0.482, 0.447])
std = np.array([0.202, 0.199, 0.201])

x_normalized_train = (x_train - mean) / std
x_normalized_test = (x_test - mean) / std

data_augmentation = models.Sequential([
    layers.ZeroPadding2D(padding=4),   # pour éviter perte d’info
    layers.RandomCrop(28, 28),
    layers.Resizing(32, 32)  # pour garder compatibilité avec le modèle
])

# =========================
# Construction du modèle
# =========================

model = models.Sequential([
    
    layers.Input(shape=(32,32,3)),

    data_augmentation,

    layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (5,5), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (5,5), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(1000, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
# =========================
# Compilation
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Entraînement
# =========================

history = model.fit(
    x_normalized_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_normalized_test, y_test)
)

# =========================
# Evaluation
# =========================

test_loss, test_acc = model.evaluate(x_normalized_test, y_test)

print("Test accuracy :", test_acc)

# =========================
# Visualisation
# =========================

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()