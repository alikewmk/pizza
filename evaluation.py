from ann import ANN
from data_handler import DataHandler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import collections

TRAIN_FILENAME = "../data/train.json"
TEST_FILENAME  = "../data/test.json"

# generate result for online evaluation
def online_eval():
    # evaluation code for online test
    handler = DataHandler()
    data = handler.generate_data(TRAIN_FILENAME)
    testing_data = handler.generate_data(TEST_FILENAME, 'test')
    ann = ANN(9,10,1)
    for i in range(80):
        print(i+1)
        ann.train(data, 5000)

    result = ann.test_without_true_label(testing_data, 0.23)
    handler.write_to_result(TEST_FILENAME, result)

# my evaluation
# add in cross validation
def my_eval(ann, training_data, testing_data, iter_num=60):
    for i in range(iter_num):
        ann.train(training_data, 5000)
    result = ann.test(testing_data, 0.34)

    return result

# for each test, use data with 100 true label and 300 false label.
def split_data(data, offset):
    train_data = []
    test_data  = []

    zero_count = 0
    one_count = 0
    for i in range(len(data)):
        if i < offset:
            train_data.append(data[i])
        else:
            if (data[i][1][0] == 1 and one_count < 100):
                test_data.append(data[i])
                one_count += 1
            else:
                train_data.append(data[i])

            if (data[i][1][0] == 0 and zero_count < 300):
                test_data.append(data[i])
                zero_count += 1
            else:
                train_data.append(data[i])

    y=collections.Counter([val[1][0] for val in test_data])
    print(y)

    return train_data, test_data

def iteration_test(data):
    train_data, test_data = split_data(data, 0)
    vals = []
    for i in range(10, 101, 10):
        print(i)
        ann = ANN(9,7,1)
        vals.append(my_eval(ann, train_data, test_data, i))

    plot_iter(vals)

def node_test(data):

    train_data, test_data = split_data(data, 0)
    result = []

    for i in range(1, 11):
        array = [9]
        array.append(i)
        array.append(1)
        print("Testing...")
        print(array)
        vals = []
        for i in range(5):
            ann = ANN(*array)
            vals.append(my_eval(ann, train_data, test_data))
        print(max(vals))
        result.append(max(vals))

    plot_node(result)

def layer_test(data):

    train_data, test_data = split_data(data, 0)
    result = []

    array = [9, 1]
    for i in range(1, 4):
        array.pop()
        array.append(10)
        array.append(1)
        print("Testing...")
        print(array)
        vals = []
        ann = ANN(*array)
        vals.append(my_eval(ann, train_data, test_data))
        print(max(vals))
        result.append((array, max(vals)))

    print(result)

def cross_validation(fold, offset, data):
    for i in range(fold):
        train_data, test_data = split_data(data, offset*i)
        ann = ANN(9,7,1)
        my_eval(ann, train_data, test_data)

# plots iteration number figure
def plot_iter(result):
    width = 10
    x = np.arange(10, 101, 10);
    plt.ylim(0.6, 0.68)
    plt.ylabel('Precision')
    plt.xlabel('Iteration')
    plt.bar(x, [val for val in result], width, color='#ababab')
    plt.show()

# plot node number in hidden layer figure
def plot_node(result):
    width = 0.5
    x = np.arange(1, 11, 1);
    plt.ylim(0.62, 0.66)
    plt.ylabel('Precision')
    plt.xlabel('Hidden Layer Node Number')
    plt.bar(x, [val for val in result], width, color="#ababab")
    plt.show()

if __name__ == "__main__":

    handler = DataHandler()
    data = handler.generate_data(TRAIN_FILENAME)
    iteration_test(data)
    node_test(data)
    layer_test(data)
    cross_validation(5, 500, data)
    online_eval()
