import numpy as np
import random

class ANN:

    def __init__(self, *nums):

        # initial node num for each layer
        # remember to add bias node to input
        self.layers = len(nums)
        self.nums = list(nums)
        self.nums[0] += 1

        # init weights of input and hidden layer with random small values
        self.weights = []
        for i in range(self.layers-1):
            self.weights.append(self.init_matrix(self.nums[i], self.nums[i+1]))

        # init x values, delta values
        self.x_vals = []
        self.deltas = []
        for i in range(self.layers):
            self.x_vals.append([0]*self.nums[i])
            self.deltas.append([0]*self.nums[i])

    # Notice that weight have to be initialized with a small number
    # positive or negative, the setting of weight affect
    # the training A LOT!!!
    def init_matrix(self, row, column):
        result = []
        for i in range(row):
            result.append([random.randint(-5,5)*0.1]*column)

        return result

    # sigmoid function
    def g_(self, z):
        return 1/(1 + np.exp(-z))

    # derivation of sigmoid function
    def dg_(self, z):
        return self.g_(z) * (1 - self.g_(z))

    def forward_prop(self, data_vec):
        # initialize input
        self.x_vals[0][:-1] = data_vec
        # set bias to one
        self.x_vals[0][-1]  = 1

        # update each column of weight array
        # notice the order of iterating layers
        for i in range(1, self.layers):
            # post layer
            for j in range(self.nums[i]):
                z = 0
                # prior layer
                for k in range(self.nums[i-1]):
                    z += self.x_vals[i-1][k]*self.weights[i-1][k][j]
                self.x_vals[i][j] = self.g_(z)

    def back_prop(self, y, alpha=0.1):
        # error on output layer
        for i in range(self.nums[-1]):
            self.deltas[-1][i] = -(y[i] - self.x_vals[-1][i])

        # error on each hidden layer
        for i in reversed(range(1, self.layers-1)):
            # prior layer
            for j in range(self.nums[i]):
                error = 0
                # post layer
                for k in range(self.nums[i+1]):
                    error += self.deltas[i+1][k]*self.weights[i][j][k]
                self.deltas[i][j] = error*self.dg_(self.x_vals[i][j])

        # update weights
        for i in reversed(range(self.layers-1)):
            for j in range(self.nums[i]):
                for k in range(self.nums[i+1]):
                    self.weights[i][j][k] -= alpha*self.deltas[i+1][k]*self.x_vals[i][j]

    # online learning
    def train(self, items, iter_num):
        len_items = len(items)
        for i in range(1, iter_num+1):
            item = items[i%len_items]
            self.forward_prop(item[0])
            self.back_prop(item[1])

    # prediction with labeled data, in order to calculate accuracy
    def test(self, items, threshold=0.5):
        len_items = len(items)
        right_count = 0
        for i in range(len_items):
            self.forward_prop(items[i][0])
            expected_val = [1] if self.x_vals[-1][0] > 0.5 else [0]
            if expected_val == items[i][1]:
                right_count += 1

        print("Accuracy: " + str(right_count/len_items))

    # prediction with unlabeled test data
    def test_without_true_label(self, items, threshold=0.5):
        len_items = len(items)
        result = []
        for i in range(len_items):
            self.forward_prop(items[i])
            print(self.x_vals[-1][0])
            result.append(1 if self.x_vals[-1][0] > threshold else 0)

        return result

### TEST
if __name__ == "__main__":

    ## TEST DATA
    data = [ [[1,1,1],[1]], [[1,1,1],[1]], [[0,0,0],[0]], [[0,0,0],[0]], [[0,0,0],[0]], [[0,0,0],[0]], [[0,0,0],[0]], [[1,1,1],[1]], [[1,1,1],[1]]]
    data_without_label = [[1,1,1], [1,1,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [1,1,1], [1,1,1]]

    ann = ANN(3,9,9,1)
    ann.train(data, 1000)
    ann.test(data)
    print(ann.test_without_true_label(data_without_label))
