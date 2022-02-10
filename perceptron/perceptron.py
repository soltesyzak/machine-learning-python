import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

value = sigmoid(4)
print(value)