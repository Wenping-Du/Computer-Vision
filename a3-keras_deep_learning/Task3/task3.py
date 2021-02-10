import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


class XORnet:
    def __init__(self, x, y, h_size):
        # Inputs:
        # x : the inputs
        # y : the groundtruth outputs
        # h_size : the number of nuerons in the hidden layer

        # we store x and y locally so we do not have to pass them everytime
        self.input = x
        self.y = y

        # W1 has a size of (3 x h_size)
        self.W1 = np.random.rand(self.input.shape[1], h_size)

        # W2 has a size of (h_size x 1)
        self.W2 = np.random.rand(h_size, 1)

        self.output = np.zeros(self.y.shape)  # This is y_hat (the output)

    def forward(self):
        # TODO:
        # implement the forward function that takes through each layer and
        # the corresponding activation function, this will generate the
        # output that should be stored in self.output
        a2 = sigmoid(np.dot(sigmoid(np.dot(X, self.W1)), self.W2))
        self.output = a2
        return np.dot((self.y - self.output).T, (self.y - self.output))

    def backward(self):
        # TODO:
        # apply the chain rule to find derivative of the loss function
        # with respect to W2 and W1
        s1 = np.dot(X, self.W1)
        y1 = sigmoid(s1)
        s2 = np.dot(y1, self.W2)

        dy2 = -2 * (self.y - self.output)
        d_W2 = - np.dot(y1.T, sigmoid_derivative(s2) * dy2)  # Replace with correct derivative of the loss wrt W2

        ds1 = sigmoid_derivative(s1) * np.dot(sigmoid_derivative(s2) * dy2, self.W2.T)
        d_W1 = - np.dot(self.input.T, ds1)  # Replace with correct derivative of the loss wrt W1

        # we update the weights with the computed derivatives
        # You do not need to edit this part
        self.W2 += d_W2
        self.W1 += d_W1


num_iterations = 1500
num_hidden_neurons = 1

# The four possible combination of two bits go as input,
# along with a bias which is always set to 1
# The input is organised as [A, B, bias]
# Each row represents a sample

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1.]])

# The corresponding output for each case

Y = np.array([[0],
              [1],
              [1],
              [0.]])

# This defines our XORnet and
net = XORnet(X, Y, num_hidden_neurons)

# We store losses after each epoch here
losses = np.zeros((num_iterations, 1))

for i in range(num_iterations):
    loss = net.forward()
    losses[i] = loss
    net.backward()

print("Expected Output: \n", (Y.T))
print("Current output :\n", net.output.T)

plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('hidden_neurons = 1')
plt.show()

