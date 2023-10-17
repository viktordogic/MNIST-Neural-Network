import numpy as np
import mnist_loader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Network:
    def __init__(self, *args):
        self.i = 0
        self.learning_rate = 0.2
        training_data_zip, validation_data_zip, testing_data_zip = mnist_loader.load_data_wrapper()
        self.training_data_list, self.validation_data_list, self.testing_data_list = list(training_data_zip), \
                                                                                     list(validation_data_zip), \
                                                                                     list(testing_data_zip)

        self.training_data_list = np.array(self.training_data_list, dtype="object")
        self.validation_data_list = np.array(self.validation_data_list, dtype="object")
        self.testing_data_list = np.array(self.testing_data_list, dtype="object")
        self.layers = np.array(args)
        self.layers = np.insert(self.layers, 0, len(self.training_data_list[0][0]))
        self.layers = np.append(self.layers, len(self.training_data_list[0][1]))

        self.first_layer = np.zeros(len(self.training_data_list[0][0]))

        self.weights = []
        for iteration in range(len(self.layers) - 1):
            self.weights.append(
                np.random.normal(0, 0.1, (self.layers[iteration] + 1) * self.layers[iteration + 1]).reshape(
                    self.layers[iteration + 1], self.layers[iteration] + 1))
        self.weights = np.array(self.weights, dtype=object)

    def output_calculation(self, index, input_set, curr_out):
        self.first_layer = input_set[index][0]
        each_sample_data, each_sample_label = input_set[index]
        each_sample_data = each_sample_data
        each_sample_data = np.insert(each_sample_data, 0, 1, axis=0)
        curr_out[0] = sigmoid(self.weights[0] @ each_sample_data)
        curr_out[0] = np.insert(curr_out[0], 0, 1)
        for layer in range(len(curr_out) - 1):
            curr_out[layer + 1] = sigmoid(self.weights[layer + 1] @ curr_out[layer])
            if layer != len(curr_out) - 2:
                curr_out[layer + 1] = np.insert(curr_out[layer + 1], 0, 1)

        self.i += 1
        return curr_out, each_sample_label

    def feeding_forward(self, input_set):
        curr_out = []
        for x in range(len(self.layers) - 1):
            temp = np.zeros(self.layers[x + 1])
            curr_out.append(temp)

        curr_out = np.array(curr_out, dtype=object)

        for index in range(input_set.shape[0]):
            self.back_propagation(self.output_calculation(index, input_set, curr_out))

    def back_propagation(self, asd):
        current_output, desired_output = asd
        deltas = []
        for x in range(len(self.layers[1:])):
            temp = np.zeros(self.layers[x + 1])
            deltas.append(temp)

        deltas = np.array(deltas, dtype=object)

        for index, output_node in enumerate(current_output[-1]):
            deltas[-1][index] = output_node * (1 - output_node) * (output_node - desired_output[index])  # III

        for layer in range(len(deltas) - 2, -1, -1):
            transpose_try = self.weights[layer + 1].T[1:]
            for index in range(len(deltas[layer])):
                sum = transpose_try[index] @ deltas[layer + 1]

                deltas[layer][index] = current_output[layer][index + 1] * (1 - current_output[layer][index + 1]) * sum  # IV

        matrix_for_first_layer = np.zeros((len(deltas[0]), len(self.first_layer)))

        ttt = np.ravel(self.first_layer)
        for svaki in range(len(deltas[0])):
            matrix_for_first_layer[svaki] = np.array(matrix_for_first_layer[svaki])
            matrix_for_first_layer[svaki] = deltas[0][svaki] * ttt

        matrix_of_next_layers = []

        for index_of_layer, deltas_of_layer in enumerate(deltas[1:]):
            zasebna_matrica = np.zeros((len(deltas[0]), len(current_output[index_of_layer])))

            for svaki in range(len(deltas_of_layer)):
                zasebna_matrica[svaki] = deltas_of_layer[svaki] * current_output[index_of_layer]
            matrix_of_next_layers.append(zasebna_matrica)

        for layer in range(len(self.weights)):
            for index, connected_node in enumerate(self.weights[layer]):
                if layer == 0:
                    connected_node[0] -= self.learning_rate * deltas[layer][index]
                    connected_node[1:] -= self.learning_rate * matrix_for_first_layer[index]
                else:
                    connected_node -= self.learning_rate * matrix_of_next_layers[layer - 1][index]

    def test(self, data):
        curr_out = []
        for x in range(len(self.layers) - 1):
            temp = np.zeros(self.layers[x + 1])
            temp = np.insert(temp, 0, 1)
            curr_out.append(temp)

        curr_out = np.array(curr_out, dtype=object)
        correct = 0
        for index in range(data.shape[0]):
            result = self.output_calculation(index, data, curr_out)

            if np.argmax(result[0][-1]) == result[1]:
                correct += 1

        print(f"{correct} of 10000 are correct. That is a percentage of {correct * 100 / 10000}%.")


if __name__ == '__main__':
    net = Network(15)
    for x in range(10):

        net.feeding_forward(net.training_data_list)
        net.test(net.testing_data_list)

