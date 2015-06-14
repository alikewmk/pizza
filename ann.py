import numpy as np
import random

TRAINING_DATA = [([1, 0], 1), ([0, 1], 1), ([1, 1], 0), ([0, 0], 0)]
TEST_DATA = [[0,0], [1,1], [1,0], [0,1]]


# Don't forget about bias
class ANN:

    def __init__(self, input_num, hidden_num, output_num):

        # add bias node to input
        self.input_num = 1 + input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

        # init weights of input and hidden layer with random small values
        self.input_weights = init_matirx(self.input_num, self.hidden_num)
        self.hidden_weights = init_matirx(self.hidden_num, self.output_num)



    # sigmoid function
    def g_(z):
        return 1/(1 + np.exp(-z))

    # derivation of sigmoid function
    def dg_(z):
        self.g_(z) * (1 - self.g_(z))

    def init_matrix(row, column):
        result = []
        for i in range(row):
            result.append([random.random*0.1]*column)

        return result





