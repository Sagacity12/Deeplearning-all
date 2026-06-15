import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Input, Conv2D, UpSampling2D, Dropout

import numpy as np
import matplotlib.pyplot as plt


input_layer = Input(shape=(28, 28, 1))

conv_layer = Conv2D(filters = 32, kernel_size= (3, 3), activation='relu', padding='same')(input_layer)

dropout_layer = Dropout(0.5)(conv_layer)

transpose_conv_layer = Conv2DTranspose(filters = 1, kernel_size= (3, 3),  padding='same')(conv_layer)

model = Model(inputs=input_layer, outputs=transpose_conv_layer)

model.compile(optimizer='adam', loss='mse')

X_train = np.random.rand(1000, 28, 28, 1)
y_train = X_train

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

X_test = np.random.rand(200, 28, 28, 1)
y_test = X_test

loss = model.evaluate(X_test, y_test)

print(f'Test loss: {loss}')

y_pred = model.predict(X_test)

n = 10

plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()

