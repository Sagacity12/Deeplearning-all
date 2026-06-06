import tensorflow as tf
from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization

import  numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Define the input layer
input_layer = Input(shape=(20,))
print(input_layer)

# Adding Hidden Layers
hidden_layer_1 = Dense(64, activation='relu')(input_layer)
hidden_layer_2 = Dense(64, activation='relu')(hidden_layer_1)

# Define the Output Layer
# suitable for binary classification
output_layer = Dense(1, activation='sigmoid')(hidden_layer_2)

# create the model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

# evaluate the model
X_test = np.random.rand(200, 20)
y_test = np.random.randint(2, size=(200, 1))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}; Test accuracy: {accuracy}")

# Dropout Layers
# Dropout is a regularization technique that helps prevent overfitting in neural networks.
# During training, Dropout randomly sets a fraction of input units to zero at each update cycle.
# This prevents the model from becoming overly reliant on any specific neurons,
# which encourages the network to learn more robust features that generalize better to unseen data.

# Key points:
# Dropout is only applied during training, not during inference.
# The dropout rate is a hyperparameter that determines the fraction of neurons to drop.

# Batch Normalization
# Batch Normalization is a technique used to improve the training stability and speed of neural networks.
# It normalizes the output of a previous layer by re-centering and re-scaling the data,which helps in stabilizing the learning process.
# By reducing the internal covariate shift (the changes in the distribution of layer inputs),
# batch normalization allows the model to use higher learning rates, which often speeds up convergence.

# Key Points:
# Batch normalization works by normalizing the inputs to each layer to have a mean of zero and a variance of one.
# It is applied during both training and inference, although its behavior varies slightly between the two phases.
# Batch normalization layers also introduce two learnable parameters that allow the model to scale and -
# shift the normalized output, which helps in restoring the model's representational power.

# Define the input layer
input_layer = Input(shape=(20 ,))

# Adding a hidden layer
hidden_layer = Dense(64, activation='relu')(input_layer)

# Adding a Dropout layer
dropout_layer = Dropout(rate=0.5)(hidden_layer)

# Adding another hidden layer after Dropout
hidden_layer_2 = Dense(64, activation='relu')(dropout_layer)

# Define the output Layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer_2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Summary of the model
model.summary()

# Adding the BatchNormalization
# Define the input layer
input_layer = Input(shape=(20,))

# Add a hidden layer
hidden_layer = Dense(64, activation='relu')(input_layer)

# Add a BatchNormalization layer
batch_norm_layer = BatchNormalization()(hidden_layer)

# Add another hidden layer after BatchNormalization
hidden_layer2 = Dense(64, activation='relu')(batch_norm_layer)

# Define the output layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Summary of the model
model.summary()