import numpy as np
import random

TRAINING_DATA = [([1, 0], 1), ([0, 1], 1), ([1, 1], 0), ([0, 0], 0)]
TEST_DATA = [[0,0], [1,1], [1,0], [0,1]]

def g_(z):
    return 1/(1 + np.exp(-z))

# Don't forget about bias
class ANN:

    # the init method need adaption
    # stop using magic numbers
    def __init__(self):
        # layer number of neural network
        self.layer = 3
        # input of each node
        self.x_matrix = [np.zeros(3), np.zeros(3), np.zeros(1)]
        # delta of each node, for back propagation
        self.d_matrix = [np.zeros(3), np.zeros(3), np.zeros(1)]
        # weight of each node, make an array of numpy array for weight
        self.w_matrix = [np.array([np.zeros(3), np.zeros(3), np.zeros(3)]), np.array([np.zeros(1), np.zeros(1), np.zeros(1)])]

    # (1) forward backward propagation
    # (2) online gradient descent
    # maybe need add self to variable
    def train(self, iter_num, items, alpha=0.1):
        len_items = len(items)
        for idx in range(iter_num):
            item = items[idx%len_items-1]

            #### FORWARD ###
            # initialize first layer
            len_xm = len(self.x_matrix)
            self.x_matrix[0] = [1] + item[0]

            ## calculate x value for each node in the next layer
            ## this part is tricky
            #
            # iterate through each layer
            # notice that because the init of first layer is special so we skip it
            # but we still need to do calculation based on its value
            # also notice that weight have one layer less than node layer
            for l in range(1, len_xm):
                # iterate through each node in current layer
                len_l = len(self.x_matrix[l])

                for n in range(len_l):
                    #print(n)
                    # here the 1 is the bias term, maybe lose it??
                    z = 1 + np.dot(self.x_matrix[l-1], self.w_matrix[l-1][:,n])
                    self.x_matrix[l][n] = g_(z)
            print("X matrix---------------")
            print(self.x_matrix)

            #### BACKWARD ###
            # notice that because the delta initialization of last layer is special, so we skip it
            y = item[1]
            self.d_matrix[-1] = np.array([-sum(np.array([y]) - self.x_matrix[-1])])

            # iterate through each layer
            for l in reversed(range(len_xm-1)):
                #print(l)
                len_l = len(self.x_matrix[l])
                for n in range(len_l):
                    # here the 1 is the bias term, maybe lose it??
                    #print(self.w_matrix[l][n])
                    #print(self.d_matrix[l+1])
                    self.d_matrix[l][n] = (1 + np.dot(self.w_matrix[l][n], self.d_matrix[l+1]))*self.x_matrix[l][n]*(1-self.x_matrix[l][n])

            print("D matrix---------------")
            print(self.d_matrix)

            ### update weights via online gradient descent
            for l in range(0, len_xm):
                len_l = len(self.x_matrix[l])
                for n in range(len_l-1):
                    #print(l)
                    for m in range(len(self.w_matrix[l][n])):
                        #print(self.w_matrix[l][n])
                        self.w_matrix[l][n][m] = self.w_matrix[l][n][m] - alpha*self.d_matrix[l][n]*self.x_matrix[l][m]

            print("W matrix---------------")
            print(self.w_matrix)


    def predict(self, items):
        result = []
        for item in items:
            print(item)
            #### FORWARD ###
            # initialize first layer
            len_xm = len(self.x_matrix)
            self.x_matrix[0] = [1] + item

            ## calculate x value for each node in the next layer
            ## this part is tricky
            #
            # iterate through each layer
            # notice that because the init of first layer is special so we skip it
            # but we still need to do calculation based on its value
            # also notice that weight have one layer less than node layer
            for l in range(1, len_xm):
                # iterate through each node in current layer
                len_l = len(self.x_matrix[l])

                for n in range(len_l):
                    #print(n)
                    # here the 1 is the bias term, maybe lose it??
                    z = 1 + np.dot(self.x_matrix[l-1], self.w_matrix[l-1][:,n])
                    self.x_matrix[l][n] = g_(z)

            result.append(self.x_matrix[-1])



        return result


    # TODO: write a method to dump model

ann = ANN()
ann.train(10, TRAINING_DATA)
print(ann.predict(TEST_DATA))
