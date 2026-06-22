import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
# Suppress all Python warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model




# Load the MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Normalize the pixel values to the range [-1, 1]
x_train = x_train.astype('float32') / 127.5 - 1.
x_train = np.expand_dims(x_train, axis=-1)

# Print the shape of the data
print(x_train.shape)


# Define the generator model
# Explanation:
# This step involves building the generator model for the GAN.
# The generator takes a random noise vector as an input and generates a synthetic image.
# The model uses Dense, LeakyReLU, BatchNormalization, and Reshape layers to achieve this.

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

# Build the generator
generator = build_generator()
generator.summary()

# Building the discriminator model
# This building the discriminator model for the GAN.
# The discriminator takes an image as an input and outputs a probability indicating whether the image is real or fake.
# The model uses Flatten, Dense, and LeakyReLU layers to achieve this.
# Objective:
# Construct the discriminator model for the GAN using the Keras functional API.
# Instructions:
# Define the discriminator.
#
# Create a Sequential model.
# Add Flatten, Dense, and LeakyReLU layers to build the discriminator.
# Compile the discriminator.
#
# Compile the model using binary cross-entropy loss and the Adam optimizer.

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator.summary()


# Building the GAN model
# This step involves combining the generator and discriminator models to form the GAN.
# The GAN takes a random noise vector as an input and generates a synthetic image.
# The discriminator is then used to determine whether the generated image is real or fake.
# The model uses the generator and discriminator to achieve this.
# Objective:
# Construct the GAN model using the Keras functional API.
# Instructions:
# Define the GAN model.
#
# Create an input layer for the noise vector.
# Pass the noise vector through the generator to produce a synthetic image.
# Pass the synthetic image through the discriminator to get the classification.
# Compile the GAN using binary cross-entropy loss and the Adam optimizer.

# Create the GAN by stacking the generator and the discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

# Build the GAN
gan = build_gan(generator, discriminator)
gan.summary()


# Sync discriminator weights from trainable to non-trainable in GAN
gan.layers[2].set_weights(discriminator.get_weights())


# Training the GAN
# Define training parameters.
#
# Set the batch size, number of epochs, and sample interval.
# Train the discriminator.
#
# Sample a batch of real images from the dataset.
# Generate a batch of synthetic images from the generator.
# Train the discriminator on both real and generated images.
# Train the generator.
#
# Generate a batch of noise vectors.
# Train the GAN to improve the generator’s ability to fool the discriminator.
# Print the progress:
#
# Print the discriminator and generator losses at regular intervals.

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and recompile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator.summary()


# Training parameters

batch_size = 64
epochs = 200
sample_interval = 10

# Adversarial ground truths
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# Training loop
for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_images, real)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, real)

    # Print the progress
    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}] [D accuracy: {100 * d_loss[1]}%] [G loss: {g_loss}]")

# Assessing the Quality of the images

# Objective:
# Evaluate the performance of the trained GAN.
# Evaluating the GAN
# After training the GAN, we need to assess the quality of the synthetic images generated by the generator.
# There are two main ways to evaluate the performance of GANs: qualitative assessment and quantitative assessment.
#
# Qualitative Assessment: Visual Inspection
# Visual inspection is a straightforward method to assess the quality of images generated by a GAN.
# You can use the sample_images function provided in the lab to visualize a grid of generated images.
# During visual inspection, look for the following qualities:
# Clarity: The images should be sharp and not blurry. Blurry images indicate that the generator is struggling to learn the patterns in the data.
# Coherence: The generated images should have a coherent structure that resembles the original images in the dataset. For example, in the case of MNIST, the generated images should resemble handwritten digits with the correct number of strokes and shapes.
# Diversity: There should be a variety of images generated by the GAN. If all images look similar, it might indicate that the generator is overfitting or has collapsed to a single mode.
# Instructions:
# Run the sample_images function after training the GAN to display a grid of generated images.
# Inspect the images for clarity, coherence, and diversity.


def sample_images(generator, epoch, num_images=25):
    noise = np.random.normal(0, 1, (num_images, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    count = 0

    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.show()

# Sample images at the end of training
sample_images(generator, epochs)


# 2. Quantitative Assessment: Metrics
# While visual inspection provides an intuitive understanding of the GAN’s performance, it can be subjective.
# To objectively evaluate GAN performance, you can use quantitative metrics such as:
#
# Inception Score (IS): This score measures both the quality and diversity of generated images by using a pre-trained classifier (such as Inception-v3) to predict the class of each image.
# A higher score indicates that the images are both high-quality and diverse.
# However, IS is not very effective for simple datasets like MNIST; it’s more suitable for complex datasets.
#
# Fréchet Inception Distance (FID): This metric calculates the distance between the distributions of generated images and real images.
# A lower FID score indicates that the generated images are more similar to real images.
# FID is commonly used and considered a reliable metric for evaluating GAN performance.
#
# Discriminator Accuracy: During training, if the discriminator's accuracy is around 50%, it suggests that the generator is producing realistic images that are hard to distinguish from real ones.
# This metric is easy to implement and provides quick feedback on the training progress.

# Calculate and print the discriminator accuracy on real vs. fake images
noise = np.random.normal(0, 1, (batch_size, 100))
generated_images = generator.predict(noise)

# Evaluate the discriminator on real images
real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
d_loss_real = discriminator.evaluate(real_images, np.ones((batch_size, 1)), verbose=0)

# Evaluate the discriminator on fake images
d_loss_fake = discriminator.evaluate(generated_images, np.zeros((batch_size, 1)), verbose=0)

print(f"Discriminator Accuracy on Real Images: {d_loss_real[1] * 100:.2f}%")
print(f"Discriminator Accuracy on Fake Images: {d_loss_fake[1] * 100:.2f}%")

# 3. Combining Qualitative and Quantitative Assessments
# For a comprehensive evaluation of the GAN:
#
# Start with visual inspection to get a quick sense of image quality.
# If the images look blurry or too similar, it might indicate problems with the training process.
#
# Use quantitative metrics like FID or discriminator accuracy to provide objective evidence of the GAN’s performance.
#
# Monitor training progress by visualizing the generator and discriminator losses over time.
# This helps in understanding if the GAN is suffering from instability or if one model is overpowering the other.