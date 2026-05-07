from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    LeakyReLU
)

# =========================================================
# Chargement du dataset
# =========================================================

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

# =========================================================
# Affichage des dimensions
# =========================================================

print("Training data shape :", train_X.shape, train_Y.shape)
print("Testing data shape  :", test_X.shape, test_Y.shape)

# =========================================================
# Classes
# =========================================================

classes = np.unique(train_Y)
nClasses = len(classes)

print("Total number of outputs :", nClasses)
print("Output classes :", classes)

# =========================================================
# Affichage de quelques images
# =========================================================

plt.figure(figsize=[5, 5])

# Première image train
plt.subplot(121)
plt.imshow(train_X[0], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Première image test
plt.subplot(122)
plt.imshow(test_X[0], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

plt.show()

# =========================================================
# Reshape
# =========================================================

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

print("After reshape:")
print(train_X.shape)
print(test_X.shape)

# =========================================================
# Normalisation
# =========================================================

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X = train_X / 255.0
test_X = test_X / 255.0

# =========================================================
# One Hot Encoding
# =========================================================

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print("Original label :", train_Y[0])
print("After one-hot :", train_Y_one_hot[0])

# =========================================================
# Split Train / Validation
# =========================================================

train_X, valid_X, train_label, valid_label = train_test_split(
    train_X,
    train_Y_one_hot,
    test_size=0.2,
    random_state=13
)

print("Train :", train_X.shape, train_label.shape)
print("Valid :", valid_X.shape, valid_label.shape)

# =========================================================
# Hyperparamètres
# =========================================================

batch_size = 64
epochs = 20
num_classes = 10

# =========================================================
# Construction du modèle
# =========================================================

fashion_model = Sequential()

# ---------------------------------------------------------
# Bloc 1
# ---------------------------------------------------------

fashion_model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation='linear',
        input_shape=(28, 28, 1),
        padding='same'
    )
)

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )
)

# ---------------------------------------------------------
# Bloc 2
# ---------------------------------------------------------

fashion_model.add(
    Conv2D(
        64,
        kernel_size=(3, 3),
        activation='linear',
        padding='same'
    )
)

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )
)

# ---------------------------------------------------------
# Bloc 3
# ---------------------------------------------------------

fashion_model.add(
    Conv2D(
        128,
        kernel_size=(3, 3),
        activation='linear',
        padding='same'
    )
)

fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )
)

# =========================================================
# Flatten
# =========================================================

fashion_model.add(Flatten())

# =========================================================
# Dense
# =========================================================

fashion_model.add(Dense(128, activation='linear'))

fashion_model.add(LeakyReLU(alpha=0.1))

# =========================================================
# Output Layer
# =========================================================

fashion_model.add(Dense(num_classes, activation='softmax'))

# =========================================================
# Compilation
# =========================================================

fashion_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

# =========================================================
# Résumé du modèle
# =========================================================

fashion_model.summary()

# =========================================================
# Entraînement
# =========================================================

fashion_train = fashion_model.fit(
    train_X,
    train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)

# =========================================================
# Evaluation
# =========================================================

test_eval = fashion_model.evaluate(
    test_X,
    test_Y_one_hot,
    verbose=0
)

print("Test loss :", test_eval[0])
print("Test accuracy :", test_eval[1])

# =========================================================
# Historique
# =========================================================

acc = fashion_train.history['accuracy']
val_acc = fashion_train.history['val_accuracy']

loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']

epochs_range = range(len(acc))

# =========================================================
# Accuracy Graph
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    acc,
    'bo',
    label='Training accuracy'
)

plt.plot(
    epochs_range,
    val_acc,
    'b',
    label='Validation accuracy'
)

plt.title('Training and Validation Accuracy')

plt.legend()

plt.show()

# =========================================================
# Loss Graph
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    loss,
    'bo',
    label='Training loss'
)

plt.plot(
    epochs_range,
    val_loss,
    'b',
    label='Validation loss'
)

plt.title('Training and Validation Loss')

plt.legend()

plt.show()