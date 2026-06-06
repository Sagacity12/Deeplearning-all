import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
import numpy as np

import os
os.environ['PATH'] += r';C:\Users\esaga\Downloads\Graphviz-15.0.0-win32\bin'

# Define a custom layer
# Define a custom dense layer with 32 units and ReLU activation.
class CustomDenseLayer(Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer='random_uniform',
                                     trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     trainable=True)
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

#  Integrate the custom layer into a model
# Create a Keras model using the custom layer.i

# # Define the model with Softmax in the output layer
model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10),
    Softmax()
])

# The Softmax activation function is used in the output layer for multi-class classification tasks,
# ensuring the model outputs probabilities that sum up to 1 for each class,
# which aligns with categorical cross-entropy as the loss function.
# This adjustment ensures the model is optimized correctly for multi-class classification.

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model summary before building")
model.summary()

# Building the model to show
model.build((1000, 20))
print("\nModel summary after building")
model.summary()


# Train the model
X_train = np.random.random((1000, 20))
y_train = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
model.fit(X_train, y_train, batch_size=32, epochs=10)

#  evaluate the model
X_test = np.random.random((1000, 20))
y_test = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Visualize the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


# Modify the model to include a Dropout layer
model = Sequential([
    CustomDenseLayer(64),
    Dropout(0.5),
    CustomDenseLayer(10)
])

# Recompile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model again
model.fit(X_train, y_train, epochs=10, batch_size=32)

