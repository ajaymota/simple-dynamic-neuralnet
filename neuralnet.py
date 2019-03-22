import numpy as np
import random
import pickle
import os


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def dropout(model, rate):
    for row in range(len(model)):
        for col in range(row):
            if random.uniform(0, 1) > rate:
                model[row][col] = 0

    model = model / rate

    return model


def train(X, Y, learning_rate, dropout_rate, batch_size, epochs, num_hidden_layers, hidden_nodes_per_layer, iterations):
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    Y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    np.random.seed(1)

    x_batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    y_batches = [Y[i:i + batch_size] for i in range(0, len(Y), batch_size)]

    for epoch in range(epochs):
        for x, y in zip(x_batches, y_batches):

            # randomly initialize our weights with mean 0
            edges_in = 2 * np.random.random((len(x[0]), len(x))) - 1

            edges_h = [None] * (num_hidden_layers - 1)

            for i in range(num_hidden_layers - 1):
                if i == 0:
                    edges_h[i] = 2 * np.random.random((len(x), hidden_nodes_per_layer)) - 1
                else:
                    edges_h[i] = 2 * np.random.random((hidden_nodes_per_layer, hidden_nodes_per_layer)) - 1

            edges_out = 2 * np.random.random((hidden_nodes_per_layer, len(y[0]))) - 1

            if os.path.isfile("pickle.db"):
                with open("pickle.db", "rb") as f:
                    edges_in = pickle.load(f)
                    edges_h = pickle.load(f)
                    edges_out = pickle.load(f)

            for j in range(iterations):

                # Feed forward through layers 0, 1, and 2
                layer_in = x
                layers_h = [None] * num_hidden_layers
                for i in range(num_hidden_layers):
                    if i == 0:
                        layers_h[i] = nonlin(np.dot(layer_in, edges_in))
                    else:
                        layers_h[i - 1] = dropout(layers_h[i - 1], dropout_rate)
                        layers_h[i] = nonlin(np.dot(layers_h[i - 1], edges_h[i - 1]))

                layer_out = nonlin(np.dot(layers_h[num_hidden_layers - 1], edges_out))

                # how much did we miss the target value?
                layer_out_error = y - layer_out

                if (j % 10000) == 0:
                    print("Error:" + str(np.mean(np.abs(layer_out_error))))

                # in what direction is the target value?
                # were we really sure? if so, don't change too much.
                layer_out_delta = layer_out_error * nonlin(layer_out, deriv=True)

                layers_h_error = [None] * num_hidden_layers
                layers_h_delta = [None] * num_hidden_layers

                ###################################################################################
                for i in reversed(range(num_hidden_layers)):
                    if i == (num_hidden_layers - 1):
                        layers_h_error[i] = layer_out_delta.dot(edges_out.T)
                        layers_h_delta[i] = layers_h_error[i] * nonlin(layers_h[i], deriv=True)
                    else:
                        layers_h_error[i] = layers_h_delta[i + 1].dot(edges_h[i].T)
                        layers_h_delta[i] = layers_h_error[i] * nonlin(layers_h[i], deriv=True)

                edges_out += layers_h[num_hidden_layers - 1].T.dot(layer_out_delta)
                for i in reversed(range(num_hidden_layers - 1)):
                    edges_h[i] += layers_h[i].T.dot(layers_h_delta[i + 1])

                edges_in += learning_rate * layer_in.T.dot(layers_h_delta[0])

        # print(layer_out)

        with open("pickle.db", "wb") as f:
            pickle.dump(edges_in, f)
            pickle.dump(edges_h, f)
            pickle.dump(edges_out, f)


# train(X, Y, learning_rate, dropout_rate, batch_size, epochs, num_hidden_layers, hidden_nodes_per_layer, iterations)
train(0, 0, 1, 1, 4, 5000, 3, 5, 5)

