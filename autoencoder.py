# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:53:55 2023

@author: DELL
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



#load the dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)

print(x_train)


#normalize the data

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# Encoder Model
encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu')
])

# Decoder Model
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28))
])


# Autoencoder Model
autoencoder = tf.keras.Sequential([encoder, decoder])


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')



hisotry = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))




decoded_imgs = autoencoder.predict(x_test)

print(decoded_imgs)


n = 10  # number of digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot( 2, n, i + n + 1)
    decoded_img = autoencoder.predict(x_test[i].reshape(1, 784))
    plt.imshow(decoded_img.reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()
