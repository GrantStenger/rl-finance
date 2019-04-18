# Import Dependencies
import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x #?
        self.weights1 = np.random.rand(self.input.shape[1], 9)
        self.weights2 = np.random.rand(9, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feed_forward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

if __name__ == "__main__":
    X = np.array([[1,0],
                  [0,1],
                  [0,0],
                  [0,0],
                  [-1,-1],
                  [0,0],
                  [1,-1],
                  [-1,-1]])
    y = np.array([[-1],[0],[0],[-1],[-1],[1],[1],[0]])
    nn = NeuralNetwork(X, y)

    print(X.shape)
    
    for i in range(10):
        print(nn.output)
        nn.feed_forward()
        nn.backprop()
    
