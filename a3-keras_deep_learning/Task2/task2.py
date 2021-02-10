
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
import time


class Perceptron:

    # input_size: dimension of the input including bias
    def __init__(self, input_size):

        # we store the input size because we will need it later
        self.input_size = input_size

        # weights (w) in randomly initalized to be the same size as the input
        self.w = np.random.randn(input_size, 3).reshape(input_size, 3)

        # we will store our accuracy after each iteration here
        self.history = []

        self.bias = 10

    def get_predict(self, X):
        return np.dot(X, self.w)

    def get_dw(self, X, Y):
        dw = np.zeros(self.w.shape)
        for xi, yi in zip(X, Y):
            predict = self.get_predict(xi)
            for i in range(self.w.shape[1]):
                if yi == i:
                    continue
                distance = predict[i] - predict[yi] + self.bias
                if distance > 0:
                    gt = yi[0]
                    dw[:, gt] -= xi
                    dw[:, i] += xi
        dw /= X.shape[0]
        return dw

    def train(self, X, Y, max_epochs=1000):
        # we clear history each time we start training
        self.history = []
        converged = False
        epochs = 0
        while not converged and epochs < max_epochs:
            dis_w = self.get_dw(X, Y)
            self.w -= dis_w * self.bias / (epochs + 1)
            self.compute_train_accuracy(X, Y)
            if self.history[-1] == 1:
                converged = True

            epochs += 1

        if epochs == max_epochs:
            print("Qutting: Reached max iterations")

        if converged:
            print(epochs)
            print("Qutting: Converged")

        self.plot_training_history()

    # The draw function plots all the points and our current estimate
    # of the boundary between the two classes. Point are colored according to
    # the current output of the classifier. Ground truth boundary is also
    # plotted since we know how we generated the data
    def draw(self, X):
        pl.close()
        out = np.matmul(X, self.w).squeeze()
        R = X[np.argmax(out, axis=1) == 0]
        G = X[np.argmax(out, axis=1) == 1]
        B = X[np.argmax(out, axis=1) == 2]

        pl.plot(R[:, 0], R[:, 1], 'ro', label='Class 0')
        pl.plot(G[:, 0], G[:, 1], 'go', label='Class 1')
        pl.plot(B[:, 0], B[:, 1], 'bo', label='Class 2')

        x = np.linspace(-1, 1)
        pl.plot(x, -(self.w[0, 0] / self.w[1, 0] * x + self.w[2, 0] / self.w[1, 0]), label='Estimated')
        pl.plot(x, -(self.w[0, 1] / self.w[1, 1] * x + self.w[2, 1] / self.w[1, 1]), label='Estimated')
        pl.plot(x, -(self.w[0, 2] / self.w[1, 2] * x + self.w[2, 2] / self.w[1, 2]), label='Estimated')

        pl.axis('tight')
        pl.xlim((-1, 1))
        pl.ylim((-1, 1))
        pl.legend()

        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1)

    # This computes the accuracy of our current estimate
    def compute_train_accuracy(self, X, Y):
        out = np.matmul(X, self.w)
        Y_bar = np.argmax(out, axis=1)
        sum = 0
        for y1, y2 in zip(Y, Y_bar):
            sum += y1 == y2
        accuracy = sum / Y_bar.shape[0]

        self.history.append(accuracy)
        print("Accuracy : %f " % (accuracy))
        if accuracy >= 1:
            self.draw(X)

    # Once training is done, we can plot the accuracy over time
    def plot_training_history(self):
        plt.ylim((-1.01, 1.01))
        plt.plot(np.arange(len(self.history)) + 1, np.array(self.history), '-x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('bias = 1')
        plt.show()


data = sio.loadmat('training_data.mat')
X = np.array(data['X'])

X = np.append(X.T, np.ones((X.shape[1], 1)), axis=1)
Y = np.array(data['Y'])
print('Training data shape:', X.shape)
print('Labels shape:', Y.shape)

number_of_samples = Y.shape[0]
max_number_of_epochs = 500
p = Perceptron(3)
p.train(X, Y, max_number_of_epochs)
print(p.w)
