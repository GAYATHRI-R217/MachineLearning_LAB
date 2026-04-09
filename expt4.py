import numpy as np

# Fix random output
np.random.seed(1)

# Input dataset
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# Normalization
X = X / np.amax(X, axis=0)
y = y / 100

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def derivatives_sigmoid(x):
    return x * (1 - x)

# Parameters
epoch = 5
lr = 0.1
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1

# Weights & Bias
wh = np.array([[0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7]])

bh = np.array([[0.1, 0.2, 0.3]])

wout = np.array([[0.3],
                 [0.2],
                 [0.5]])

bout = np.array([[0.1]])
# Training
for i in range(epoch):

    # Forward Propagation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)

    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad

    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

    # Update weights
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr

    # Print results
    print("-----------Epoch-", i+1, "Starts----------")
    print("Input:\n", X)
    print("Actual Output:\n", y)
    print("Predicted Output:\n", output)
    print("-----------Epoch-", i+1, "Ends----------\n")

print("Input:")
print(X)

print("Actual Output:")
print(y)

print("Predicted Output:")
print(output)