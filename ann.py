import numpy as np
import random

# Don't forget about bias
class ANN:

    def __init__(self, input_num, hidden_num, output_num):

        # add bias node to input
        self.input_num  = input_num + 1
        self.hidden_num = hidden_num
        self.output_num = output_num

        # init weights of input and hidden layer with random small values
        self.input_weights  = self.init_matrix(self.input_num, self.hidden_num)
        self.hidden_weights = self.init_matrix(self.hidden_num, self.output_num)

        # init x values
        self.input_x  = [1]*input_num + [1]
        self.hidden_x = [1]*self.hidden_num
        self.output_x = [1]*self.output_num

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
        self.input_x[:-1] = data_vec
        # set bias to one
        self.input_x[-1] = 1

        # update each column of weight array
        # notice the order of iterating layers
        for j in range(self.hidden_num):
            z = 0
            for i in range(self.input_num):
                z += self.input_x[i]*self.input_weights[i][j]
            self.hidden_x[j] = self.g_(z)

        for j in range(self.output_num):
            z = 0
            for i in range(self.hidden_num):
                z += self.hidden_x[i]*self.hidden_weights[i][j]
            self.output_x[j] = self.g_(z)

    def back_prop(self, y, alpha=0.1):
        # error on output layer
        output_del = [0]*self.output_num
        for i in range(self.output_num):
            output_del[i] = -(y[i] - self.output_x[i])

        # error on hidden layer
        # notice that the order of iterating layer is reversed
        hidden_del = [0]*self.hidden_num
        for i in range(self.hidden_num):
            error = 0
            for j in range(self.output_num):
                error += output_del[j]*self.hidden_weights[i][j]
            hidden_del[i] = error * self.dg_(self.hidden_x[i])

        # update hidden weight
        for i in range(self.hidden_num):
            for j in range(self.output_num):
                self.hidden_weights[i][j] -= alpha*output_del[j]*self.hidden_x[i]

        # update input weight
        for i in range(self.input_num):
            for j in range(self.hidden_num):
                self.input_weights[i][j] -= alpha*hidden_del[j]*self.input_x[i]

    # online learning
    def train(self, items, iter_num):
        len_items = len(items)
        for i in range(1, iter_num+1):
            item = items[i%len_items]
            self.forward_prop(item[0])
            self.back_prop(item[1])

    def test(self, items):
        len_items = len(items)
        right_count = 0
        for i in range(len_items):
            self.forward_prop(items[i][0])
            expected_val = [1] if self.output_x[0] > 0.5 else [0]
            #print(self.output_x)
            #print(items[i][1])
            #print(expected_val)
            if expected_val == items[i][1]:
                right_count += 1

        print("Accuracy: " + str(right_count/len_items))

    def test_without_true_label(self, items):
        len_items = len(items)
        result = []
        for i in range(len_items):
            self.forward_prop(items[i])
            print(self.output_x[0])
            result.append(1 if self.output_x[0] > 0.22 else 0)

        return result

### TEST
if __name__ == "__main__":

    ## TEST DATA
    data = [ [[1,1,1],[1]], [[1,1,1],[1]], [[0,0,0],[0]], [[0,0,0],[0]], [[0,0,0],[0]], [[0,0,0],[0]], [[0,0,0],[0]], [[1,1,1],[1]], [[1,1,1],[1]]]
    data_without_label = [[1,1,1], [1,1,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [1,1,1], [1,1,1]]

    ann = ANN(3,2,1)
    ann.train(data, 1000)
    ann.test(data)
    print(ann.test_without_true_label(data_without_label))
