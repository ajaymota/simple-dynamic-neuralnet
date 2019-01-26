from numpy import abs, array, dot, exp, mean, random

class NeuralNet:

    __hidden_layers = -1
    __nodes_per_layer = -1
    __input_layer = -1
    __output_layer = -1
    __learn_rate = -1

    def __init__(self, hidden_layers=4, nodes_per_layer=4, input_layer=3, output_layer=1, learn_rate=120000):
        self.__hidden_layers = hidden_layers
        self.__nodes_per_layer = nodes_per_layer
        self.__input_layer = input_layer
        self.__output_layer = output_layer
        self.__learn_rate = learn_rate

    def sigmoid(self, x, deriv=False):
        if (deriv == True):
            return x * (1 - x)

        return 1 / (1 + exp(-x))

    def main(self):
        input = array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

        output = array([[0],
                   [1],
                   [1],
                   [0]])

        random.seed(1)

        # randomly initialize our weights with mean 0
        edges_in = 2 * random.random((self.__input_layer, self.__nodes_per_layer)) - 1
        edges_h = [None] * (self.__hidden_layers-1)

        for i in range(self.__hidden_layers-1):
            edges_h[i] = 2 * random.random((self.__nodes_per_layer, self.__nodes_per_layer)) - 1

        edges_out = 2 * random.random((self.__nodes_per_layer, self.__output_layer)) - 1

        for j in range(self.__learn_rate):

            # Feed forward through layers 0, 1, and 2
            layer_in = input
            layers_h = [None] * self.__hidden_layers

            for i in range(self.__hidden_layers):
                if i == 0:
                    layers_h[i] = self.sigmoid(dot(layer_in, edges_in))
                else:
                    layers_h[i] = self.sigmoid(dot(layers_h[i-1], edges_h[i-1]))

            layer_out = self.sigmoid(dot(layers_h[self.__hidden_layers-1], edges_out))

            # how much did we miss the target value?
            layer_out_error = output - layer_out

            if (j % 10000) == 0:
                print("Error:" + str(mean(abs(layer_out_error))))

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_out_delta = layer_out_error * self.sigmoid(layer_out, deriv=True)

            layers_h_error = [None] * self.__hidden_layers
            layers_h_delta = [None] * self.__hidden_layers

            ###################################################################################
            for i in reversed(range(self.__hidden_layers)):
                if i == (self.__hidden_layers-1):
                    layers_h_error[i] = layer_out_delta.dot(edges_out.T)
                    layers_h_delta[i] = layers_h_error[i] * self.sigmoid(layers_h[i], deriv=True)
                else:
                    layers_h_error[i] = layers_h_delta[i+1].dot(edges_h[i].T)
                    layers_h_delta[i] = layers_h_error[i] * self.sigmoid(layers_h[i], deriv=True)

            edges_out += layers_h[self.__hidden_layers-1].T.dot(layer_out_delta)
            for i in reversed(range(self.__hidden_layers-1)):
                edges_h[i] += layers_h[i].T.dot(layers_h_delta[i+1])

            edges_in += layer_in.T.dot(layers_h_delta[0])


instance = NeuralNet()
instance.main()
