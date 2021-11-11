import numpy as np
from matplotlib.pyplot import plot, show


def generate_input(input_neurons, output_neurons):
    #Put your input here
    return input_layer, solution


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward(weights, bias, lay):
    num = len(weights)
    layers_raw = np.empty(num, dtype = np.ndarray)
    layers = np.empty(num + 1, dtype = np.ndarray)
    for i in range(num):
        layers[i] = lay
        output = weights[i] @ lay + bias[i]
        layers_raw[i] = output
        lay = np.maximum(output, 0)
    layers[-1] = sigmoid(output)
    return layers, layers_raw


def backpropagation(bias_gradient, weight_gradient, layers, layers_raw, weights, cost):
    # first backpropagation is special since use a different activation function
    tmp_bias = cost * der_sigmoid(layers_raw[-1])
    bias_gradient[-1] -= tmp_bias
    weight_gradient[-1] -= tmp_bias @ layers[-2].T
    cost = weights[-1].T @ tmp_bias
    for i in range(2, len(weights)):
        tmp_bias = cost * np.heaviside(layers_raw[-i], 0)
        bias_gradient[-i] -= tmp_bias
        weight_gradient[-i] -= tmp_bias @ layers[-i - 1].T
        cost = weights[-i].T @ tmp_bias

    tmp_bias = cost * np.heaviside(layers_raw[0], 0)
    bias_gradient[0] -= tmp_bias
    weight_gradient[0] -= tmp_bias @ layers[0].T


def validation(weights, bias, num_input, num_output):
    found = 0
    for n in range(val_num):
        # Put your validation code here

    print(found / val_num)


def neural_network(num_neurons: list):
    global learning_rate
    num_layers = len(num_neurons)
    if num_layers < 2:
        print("Non hai specificato abbastanza layer di neuroni")
        exit(-1)
    bias = np.empty(num_layers - 1, dtype = np.ndarray)
    weights = np.empty(num_layers - 1, dtype = np.ndarray)
    bias_gradient = np.empty(num_layers - 1, dtype = np.ndarray)
    weights_gradient = np.empty(num_layers - 1, dtype = np.ndarray)
    for i in range(num_layers - 1):
        bias[i] = (np.zeros((num_neurons[i+1], 1)))
        weights[i] = (np.random.rand(num_neurons[i+1], num_neurons[i]))
        bias_gradient[i] = (np.zeros((num_neurons[i+1], 1)))
        weights_gradient[i] = (np.random.rand(num_neurons[i+1], num_neurons[i]))

    for k in range(num_epoch):
        loss = 0
        bias_gradient = bias_gradient * 0
        weights_gradient = weights_gradient * 0
        for i in range(batch_size):
            inp, sol = generate_input(num_neurons[0], num_neurons[-1])
            layers, layers_raw = forward(weights, bias, inp)
            output = layers[-1]
            loss += np.square(output - sol).sum()

            backpropagation(bias_gradient, weights_gradient, layers, layers_raw, weights, (output - sol) * 2)

        bias += learning_rate * bias_gradient / batch_size
        weights += learning_rate * weights_gradient / batch_size

        if k % 20 == 0 and k != 0:
            loss = loss / batch_size
            history.append(loss)
            print(f'Loss: {loss} LS:{learning_rate}')
        if k % descend_step == 0 and k != 0:
            learning_rate /= 10
    return weights, bias


if __name__ == '__main__':
    num_epoch = 1000
    batch_size = 64
    learning_rate = 0.1
    descend_step = numEpoch // 2
    val_num = 1000
    history = []
    network = [8, 4, 4]
    weights, bias = neural_network(network)
    validation(weights, bias, network[0], network[-1])
    plot(range(20, numEpoch, 20), history)
    show()

