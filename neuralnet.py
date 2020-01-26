################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip, copy
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    normalized = np.divide(np.subtract(img, img.min()), img.max()-img.min())
    centered = normalized - normalized.mean()
    standardized = centered / centered.std()
    return standardized


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    # res = np.zeros((len(labels), num_classes))
    # for label in labels:
    #     res[label] = 1
    # return res 
    return np.array([[0 for a in range(0,label)]+
                    [1]+
                    [0 for b in range(label+1,num_classes)] 
                    for label in labels])


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    # raise NotImplementedError("Softmax not implemented")
    # Using this: https://stats.stackexchange.com/questions/304758/softmax-overflow
    x_reduced = np.subtract(x, x.max())
    norm_fac = np.sum(np.exp(x_reduced))
    return np.divide(np.exp(x_reduced), norm_fac).T



class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        # raise NotImplementedError("Sigmoid not implemented")
        res = np.divide(1., (np.add(1., np.exp(-x))))
        self.grad_ = np.multiply(res, np.subtract(1., res))
        return res

    def tanh(self, x):
        """
        Implement tanh here.
        """
        # raise NotImplementedError("Tanh not implemented")
        res = np.tanh(x)
        self.grad_ = np.subtract(1., np.power(res, 2))
        return res

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        # raise NotImplementedError("ReLu not implemented")
        res = np.maximum(np.zeros(x.shape), x)
        self.grad_ = np.greater(x, np.zeros(x.shape), dtype=int)
        return res

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        # raise NotImplementedError("Sigmoid gradient not implemented")
        # sigmoid(x) * (1 - sigmoid(x))
        return self.grad_

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        # raise NotImplementedError("tanh gradient not implemented")
        # 1 - (tanh(x)^2)
        return self.grad_

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        # raise NotImplementedError("ReLU gradient not implemented")
        # if x <= 0: return 0, else return 1
        return self.grad_


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.rand(in_units, out_units)    # Declare the Weight matrix
        self.b = np.ones((in_units, 1))    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        # raise NotImplementedError("Layer forward pass not implemented.")
        # Assume x is batch first
        self.x = x.reshape(np.max(x.shape), 1);
        self.a = np.matmul(self.x.T, self.w)
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # delta *
        self.d_x = np.matmul(self.w, delta)
        self.d_w = np.multiply(self.d_x, self.x)
        # TODO: Add learning rate.
        self.w = np.add(self.w, self.d_w)
        return self.d_x



class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        z = copy.deepcopy(x)
        self.x = z
        loss = targets

        for layer in self.layers:
            z = layer(z)
        self.y = softmax(z)

        if loss is not None:
            self.targets = targets
            loss = self.loss(self.y, targets)

        return self.y, loss

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        # TODO: Are we expected to convert logits here?
        return np.dot(targets.T, np.log(logits))

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = copy.deepcopy(self.y)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)



def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # x_valid, y_valid = ...

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
