import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function and its derivatives
def sigmoid(z):
    return 1/(1+np.exp(-z))

# the sigmoid gradient
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Define the ReLU function
def relu(z):
    return np.maximum(0, z)

# Define the ReLU derivative gradient
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Define the tanh function
def tanh(z):
    return np.tanh(z)
    # return np.exp(z) - np.exp(-z) / (np.exp(z) + np.exp(-z))

# tanh derivative
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

# Generate a range of input values
z = np.linspace(-5, 5, 100)

# Plot the activation functions
plt.figure(figsize=(12, 6))

# for sigmoid_grad and relu_grad
# sigmoid_grad = sigmoid_derivative(z)
# relu_grad = relu_derivative(z)

#  for relu and tanh
relu_grad = relu_derivative(z)
tanh_grad = tanh_derivative(z)

# Plot Sigmoid and its derivative
# plt.subplot(1, 2, 1)
# plt.plot(z, sigmoid(z), label='Sigmoid Activation', color='b')
# plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
# plt.title('Sigmoid Activation & Gradient')
# plt.xlabel('Input Value (z)')
# plt.ylabel('Activation / Gradient')
# plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot tanh and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, tanh(z), label='Tanh Activation', color='g')
plt.plot(z, tanh_grad, label="Tanh Derivative", color='r', linestyle='--')
plt.title('tanh Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()