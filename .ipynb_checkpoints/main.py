import numpy as np

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    network = {}
    num_nodes_previous = num_inputs

    for layer in range(num_hidden_layers + 1):
        layer_name = 'output' if layer == num_hidden_layers else f'layer_{layer+1}'
        num_nodes = num_nodes_output if layer == num_hidden_layers else num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            network[layer_name][f'node_{node+1}'] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'biases':  np.around(np.random.uniform(size=1), decimals=2)
            }
        num_nodes_previous = num_nodes
    return network

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward_propagation(network, inputs):
    layer_inputs = list(inputs)

    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for node in layer_data:
            node_data = layer_data[node]
            z = np.dot(node_data['weights'], layer_inputs) + node_data['biases']
            output = sigmoid(z)
            layer_outputs.append(np.around(output[0], decimals=4))

        if layer != 'output':
            print(f'Hidden {layer} outputs: {layer_outputs}')

        layer_inputs = layer_outputs

    print(f'Prediction: {layer_inputs}')
    return layer_inputs


net = initialize_network(2, 2, [2, 2], 1)
forward_propagation(net, [0.5, 0.85])