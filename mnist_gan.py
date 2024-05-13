from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


# Define input image dimensions
img_rows, img_cols, channels = 28, 28, 1
img_shape = (img_rows, img_cols, channels)

# Generator Network (using functional API)
def build_generator():
    noise_shape = (100,)

    model = Sequential([
        Dense(256, input_shape=noise_shape),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(np.prod(img_shape), activation='tanh'),
        Reshape(img_shape)
    ])

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)


# Disciminator Network (using functional API)
def build_discriminator():


    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def train(epochs, batch_size=128, save_interval=50):

  # Load the MNIST dataset 
  (X_train, _), (_, _) = mnist.load_data()

  # Preprocess data 
  X_train = (X_train.astype(np.float32) - 127.5) / 127.5
  X_train = np.expand_dims(X_train, axis=3)

  half_batch = batch_size // 2

  for epoch in range(epochs):

    # Train Discriminator on real and fake images
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_imgs = X_train[idx]
    noise = np.random.normal(0, 1, (half_batch, 100))
    fake_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

    # Train Generator to fool Discriminator
    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_y = np.ones((batch_size, 1))
    g_loss = combined.train_on_batch(noise, valid_y)

    #Print progress
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_loss, g_loss))

    # Save generated images
    if epoch % save_interval == 0:
      save_imgs(epoch)


# Saving images
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)


    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()





# Define optimizer
optimizer = Adam(0.0002, 0.5)  # Learning rate and momentum

# Build and compile discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build and compile generator (no metrics tracked)
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# Define input noise for generator
z = Input(shape=(100,))
img = generator(z)

# Freeze discriminator weights during generator training
discriminator.trainable = False

# Discriminator output for generated and real images
valid = discriminator(img)

# Combined model (generator + discriminator) for training generator
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Train the models
train(epochs=10000, batch_size=32, save_interval=10)

# Save generator model
generator.save('generator_model.h5')
