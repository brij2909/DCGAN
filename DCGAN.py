import warnings
warnings.filterwarnings('ignore')

## Load Libraries

import keras
from keras.layers import add
from keras.datasets import mnist
from keras.layers import *
from keras.layers import LeakyReLU
from keras.models import Sequential , Model
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.optimizers import legacy as legacy_optimizers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(X_train , _),(_,_)=mnist.load_data()
print(X_train.shape)

X_train = (X_train-127.5)/127.5

print(X_train.min())
print(X_train.max())

Total_Epochs = 50
Batch_Size = 256
Half_Batch = 128

No_of_Batches = int(X_train.shape[0]/Batch_Size)

Noise_Dim = 100

adam = legacy_optimizers.Adam(lr = 2e-4, beta_1 = 0.5)  # Using legacy optimizer

# Generator Model : Upsampling

generator = Sequential()
generator.add(Dense(units = 7*7*128, input_shape = (Noise_Dim,)))
generator.add(Reshape((7,7,128)))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())

#(7,7,128)  >  (14,14,64)

generator.add(Conv2DTranspose(64, (3,3), strides = (2,2), padding= 'same'))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())

#(14,14,64)  -->  (28,28,1)

generator.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', activation='tanh'))

generator.compile(loss = keras.losses.binary_crossentropy , optimizer=adam)

print(generator.summary())

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.layers import LeakyReLU

# Discriminator Model: Down Sampling
# (28, 28, 1) -> (14, 14, 64)
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
discriminator.add(LeakyReLU(0.2))

# (14, 14, 64) -> (7, 7, 128)
discriminator.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))

# Flatten to (7, 7, 128) -> 6272
discriminator.add(Flatten())
discriminator.add(Dense(100))
discriminator.add(LeakyReLU(0.2))

# Output layer
discriminator.add(Dense(1, activation='sigmoid'))

# Compile the model
discriminator.compile(loss=keras.losses.binary_crossentropy, optimizer=adam)

# Print model summary
discriminator.summary()

# Combined Model

discriminator.trainable = False

gan_input = Input(shape = (Noise_Dim, ))

generated_img = generator(gan_input )

gan_output = discriminator(generated_img)

# Functional API

model = Model(gan_input, gan_output)

model.compile(loss = keras.losses.BinaryCrossentropy, optimizer=adam )
model.compile(loss='binary_crossentropy', optimizer=adam)


model.summary()

X_train = X_train.reshape(-1, 28,28,1)
print(X_train.shape)

def display_images(sample=25):
    noise = np.random.normal(0,1,size=(sample, Noise_Dim))

    generated_img = generator.predict(noise)

    plt.figure(figsize=(10,10))
    for i in range(sample):
        plt.subplot(5,5,i+1)
        plt.imshow(generated_img[i].reshape(28,28), cmap='binary')
        plt.axis('off')
    
    plt.show()
## Training Loop

d_losses = []
g_losses = [ ]

for epoch in range(Total_Epochs):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0

    # Mini batch gradient descent
    for step in range(No_of_Batches):
        # Step 1: Train Discriminator
        discriminator.trainable = True

        # Get the real data
        idx = np.random.randint(0, 60000, Half_Batch)
        real_imgs = X_train[idx]

        # Get fake data
        noise = np.random.normal(0, 1, size=(Half_Batch, Noise_Dim))
        fake_imgs = generator.predict(noise)

        # Labels
        real_y = np.ones((Half_Batch, 1)) * 0.9
        fake_y = np.zeros((Half_Batch, 1))

        # Train D
        d_loss_real = discriminator.train_on_batch(real_imgs, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        epoch_d_loss += d_loss

        # Step 2: Train Generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, size=(Batch_Size, Noise_Dim))
        ground_truth_y = np.ones((Batch_Size, 1))
        g_loss = model.train_on_batch(noise, ground_truth_y)
        epoch_g_loss += g_loss

    # Print average losses for the current epoch
    print(f"Epoch {epoch+1}, Disc Loss {epoch_d_loss/No_of_Batches}, Generator Loss {epoch_g_loss/No_of_Batches}")

    # Save generator and display generated images every 10 epochs
    if (epoch+1) % 10 == 0:
        generator.save("generator.h5")
        display_images()

#### Done
