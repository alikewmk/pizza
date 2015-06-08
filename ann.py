import numpy as np
import random

TRAINING_DATA = [([1, 0], 1), ([0, 1], 1), ([1, 1], 0), ([0, 0], 0)]
TEST_DATA = [[1,1], [0,0]]

# Don't forget about bias
class ANN:

    # the init method need adaption
    # stop using magic numbers
    def __init__(self):
        # layer number of neural network
        self.layer = 3
        # input of each node
        self.x_matrix = np.array([np.zeros(2), np.zeros(2), np.zeros(1)])
        # delta of each node, for back propagation
        self.d_matrix = np.array([np.zeros(2), np.zeros(2), np.zeros(1)])
        # weight of each node
        self.w_matrix = np.array([np.zeros((2,2)), np.zeros(2)])

    def g_(z):
        return 1/(1 + np.exp(-z))

    # (1) forward backward propagation
    # (2) online gradient descent
    # maybe need add self to variable
    def train(iter_num, items, alpha=0.1):
        len_items = len(items)
        for idx in range(iter_num):
            item = items[idx%len_items-1]

            #### FORWARD ###
            # initialize first layer
            len_xm = len(x_matrix)
            x_matrix[0] = item[0]

            ## calculate x value for each node in the next layer
            ## this part is tricky
            #
            # iterate through each layer
            # notice that because the init of first layer is special so we skip it
            # also notice that weight should always have one layer less than node layer
            for l in range(1, len_xm):
                # iterate through each node in current layer
                len_l = len(x_matrix[l])
                for n in range(len_l):
                    # here the 1 is the bias term, maybe lose it??
                    z = 1 + np.dot(x_matrix[l], w_matrix[l-1][:,n])
                    x_matrix[l][n] = g_(z)

            #### BACKWARD ###
            # notice that because the delta initialization of last layer is special, so we skip it
            y = item[1]
            d_matrix[-1] = np.array([np.dot(x_matrix[-1], np.array([y]))
            # iterate through each layer
            for l in reversed(range(len_xm-1)):
                len_l = len(x_matrix[l])
                for n in range(len_l):
                    # here the 1 is the bias term, maybe lose it??
                    d_matrix[l][n] = (1 + np.dot(w_matrix[l][n], d_matrix[l+1]))*x_matrix[l][n]*(1-x_matrix[l][n])

            #### update weights via online gradient descent
            for l in range(1, len_xm):
                len_l = len(x_matrix[l])
                for n in range(len_l):
                    for m in len(w_matrix[l][n]):
                        w_matrix[l][n][m] = w_matrix[l][n][m] - alpha*d_matrix[l][n]*x_matrix[l-1][m]


    def predict(items):
        for item in items:
            #### FORWARD ###
            # initialize first layer
            len_xm = len(x_matrix)
            x_matrix[0] = item
            ## calculate x value for each node in the next layer
            ## this part is tricky
            #
            # iterate through each layer
            # notice that because the init of first layer is special so we skip it
            # also notice that weight should always have one layer less than node layer
            for l in range(1, len_xm):
                # iterate through each node in current layer
                len_l = len(x_matrix[l])
                for n in range(len_l):
                    # here the 1 is the bias term, maybe lose it??
                    z = 1 + np.dot(x_matrix[l], w_matrix[l-1][:,n])
                    x_matrix[l][n] = g_(z)

        return x_matrix[-1]


    # TODO: write a method to dump model
    #
