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
import KFold


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
    return np.divide(np.exp(x_reduced), norm_fac)


def average(x):
    """
    Helper function for average across the n samples
    Input vector should be nxd, where n is the number of samples
    """
    return (np.sum(x, axis=0)/x.shape[0]).reshape((-1, 1))


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

    def backward(self, delta, lr=None):
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
        # print(self.grad_)
        return res

    def tanh(self, x):
        """
        Implement tanh here.
        """
        # raise NotImplementedError("Tanh not implemented")
        # TODO: Why multiply by these numbers?
        # res = 1.7159*np.tanh((2/3)*x)
        res = np.tanh(x)
        self.grad_ = np.subtract(1., np.power(res, 2))
        return res

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        # raise NotImplementedError("ReLu not implemented")
        res = np.maximum(np.zeros(x.shape), x)
        self.grad_ = np.greater(x, np.zeros(x.shape), dtype=float)
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
        self.w = np.random.normal(0, 1./np.sqrt(in_units), (in_units, out_units))  # Declare the Weight matrix
        # self.w = np.random.randn(in_units, out_units)
        self.b = 0    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)
        self.in_units = in_units
        self.out_units = out_units
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = 0  # Save the gradient w.r.t b in this

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
        # Bias Initialized to 0 according to https://piazza.com/class/k53fkn2c83f53l?cid=197
        # Assume x is batch first
        self.x = x
        self.a = np.matmul(self.x, self.w)
        return self.a

    def backward(self, delta, lr=0.005):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # delta *
        # print(delta.shape, self.w.shape)

        # print(self.x.T.shape)
        # print(delta.shape)
        # self.d_w = np.matmul(self.x.T, delta)
        d_x_temp = []
        d_w_temp = np.zeros(self.w.shape)
        for i in range(128):
            
            temp = []
            for j in range(self.in_units):
                buffer = 0
                for k in range(self.out_units):
                    buffer += delta[i][k]*self.w[j][k]
                temp.append(buffer)
            d_x_temp.append(temp)  
            

            d_w_temp += np.outer(self.x.T[:,i], delta[i,:])
        
        self.d_w = d_w_temp
        self.d_x = np.array(d_x_temp)
        # print(self.d_w.shape)
        # print("--------------")
        # print("x")
        # print(self.x.T)
        # TODO: Add learning rate.
        # print( lr * self.d_w)
        # print(self.d_w, lr)
        self.w = np.add(self.w, lr * self.d_w)
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
        self.num_samples = 0 # Number of samples in input

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
        self.x = copy.deepcopy(x)
        self.targets = targets
        self.num_samples = self.x.shape[0]

        z = self.x 
        for layer in self.layers:
            z = layer(z)
        self.y = softmax(z)

        if targets is not None:
            loss = self.loss(self.y, targets) 
            return softmax(self.y), loss

        return softmax(self.y)

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        # TODO: Are we expected to convert logits (i.e. call softmax) here?
        eps = 1e-6
        # print(targets.shape, np.log(logits+eps).shape ,np.dot(targets, np.log(logits+eps).T).shape  )

        return -1. * np.sum([np.dot(targets[i], np.log(logits[i]+eps)) for i in range(targets.shape[0])])
        # return -1. * np.matmul(targets, np.log(logits+eps).T)#/self.num_samples

    def grad_loss(self, logits, targets):
        '''
        compute the gradient w.r.t y of cross-entropy loss
        '''
        # return -1. * np.dot(1./logits, targets)
        # This should also backpropagate through Softmax as well
        return np.subtract(targets, logits)

    def backward(self, lr=0.005):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        # delta = np.array([np.subtract(y, t) for y, t in zip(self.y, self.targets)])
        # print(self.y.shape)
        delta = self.grad_loss(self.y, self.targets)
        for layer in reversed(self.layers):
            # print("delta")
            # print(delta)
            delta = layer.backward(delta, lr=lr)


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    
    bs = config["batch_size"]
    # bs = x_train.shape[0]
    lr = config["learning_rate"]
    # bs = 1

    for e in range(config["epochs"]):
        loss_sum = 0.
        b_start = 0
        correct = 0
        print("----Epoch %d ---"%e)
        while b_start < x_train.shape[0]-1:
            # initial = (model.layers[0].w, model.layers[2].w,model.layers[4].w)

            b_end = min(x_train.shape[0], b_start+bs)
            # print("--- Forward - batch %d----"%(b_start))
            logits, loss = model.forward(x_train[b_start:b_end], y_train[b_start:b_end])
            # print("--- backward - batch %d----"%(b_start))
            model.backward(lr=lr)

        

            for i in range(128):
                correct += int(np.argmax(logits[i,:]) == np.argmax(y_train[b_start+i, :]))
                # correct += np.sum([np.argmax(row) for row in logits] == [ y_train[b_start:b_end]] )
            
            loss_sum += loss
            b_start += bs

            # print(model.layers[0].w, model.layers[2].w,model.layers[4].w)
            # print( model.layers[0].w - initial[0], model.layers[2].w - initial[1], model.layers[4].w - initial[2])

        #     break
        # break
        # Accuracy loss
        print(loss_sum / float(x_train.shape[0]) )
        print(correct / float(x_train.shape[0]) )



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

    for layer in model.layers:
        if isinstance(layer, Layer):
            print(layer.w.shape)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # x_valid, y_valid = ...
    val_perc = 0.2

    idxs = np.arange(x_train.shape[0])
    np.random.shuffle(idxs)
    val_end = x_train.shape[0] * (1.-val_perc)
    mask = idxs > val_end


    x_valid, y_valid = x_train[mask,:], y_train[mask]
    x_train, y_train = x_train[~mask,:], y_train[~mask]

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
