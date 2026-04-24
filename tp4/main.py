from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical 
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# reshape + normalization
X_train = X_train.reshape(50000, 3072) / 255.0
X_test  = X_test.reshape(10000, 3072) / 255.0

# one-hot encoding
Y_train = to_categorical(y_train, 10)
Y_test  = to_categorical(y_test, 10)

model = Sequential([
    Dense(1024, input_dim=3072),
    Activation('relu'),

    Dense(512),
    Activation('relu'),

    Dense(512),
    Activation('relu'),

    Dense(10),
    Activation('softmax')
])

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)

model.fit(
    X_train, Y_train,
    batch_size=100,
    epochs=100,
    validation_data=(X_test, Y_test)
)