import numpy as np

weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)
print(weights)
print(biases)

x_1 = 0.5
x_2 = 0.85
print('x1 is {} and x2 is {}'.format(x_1, x_2))

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weights sum of the input at the first node in the hidden layer is {}'.format(z_11))

z_12 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weights sum of the input at the second node in the hidden layer is {}'.format(z_12, decimals=4))

a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(a_11, decimals=4))

a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the second node in the hidden layer is {}'.format(a_12, decimals=4))

z_2 = a_11 * weights[0] + a_12 * weights[1] + biases[0]
print('The weights sum of the input at the  node in the hidden layer is {}'.format(z_2, decimals=4))

a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(a_2, decimals=4))

# Building a Neural Network
# number of inputs
n = 2
# number of hidden layers
num_hidden_layers = 2
# number of nodes in each hidden layer
m = [2, 2]
# number of nodes in the output layer
num_nodes_output = 1

# number of nodes in the previous layer
num_nodes_previous = n
# initialize network an empty dictionary
network = {}

# loop through each layer and randomly initialize the weights and biases associated with each node
# notice adding 1 to the number of hidden layers in order to include the output layer
for layer in range(num_hidden_layers + 1):

    # determine name of Layer
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]

    # initialize weights and biases associated with each node in the current layer
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node + 1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'biases': np.around(np.random.uniform(size=1), decimals=2)
        }
    num_nodes_previous = num_nodes
print(network)


def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'biases': np.around(np.random.uniform(size=1), decimals=2)
            }
        num_nodes_previous = num_nodes

    return network

def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-weighted_sum))
print('The activation of the first node in the hidden layer is {}'.format(node_activation(weights), decimals=4))


def forward_propagation(network, inputs, weighted_sum):

    layer_inputs = list(inputs)
    for layer in network:

        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]

            # compute the weighted sum and the output of each node at the same time
            compute_weights_sum = weighted_sum
            weights = node_data[compute_weights_sum]
            node_output = node_activation(weights(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs
    network_predictions = layer_outputs
    return network_predictions 