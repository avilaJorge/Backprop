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
import matplotlib.pyplot as plt


NG_IMPLEMENTATION = False

def load_config(path, fname='config.yaml'):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(fname, 'r'), Loader=yaml.SafeLoader)


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
    # Using this: https://stats.stackexchange.com/questions/304758/softmax-overflow
    def helper(row):
        x_reduced = np.subtract(row, row.max())
        norm_fac = np.sum(np.exp(x_reduced))
        return np.divide(np.exp(x_reduced), norm_fac)

    if x.ndim == 1: x.reshape((-1,1))
    ret = np.array([helper(x_i) for x_i in x])
    return ret


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

    def backward(self, delta, lr=None, lamda = None, momentum = None, momentum_gamma = None):
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

    def deepcopy(self):
        result = Activation(self.activation_type)
        if self.x is not None: result.x = self.x.copy()
        return result


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
        self.b = np.zeros((1, out_units))    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)
        self.in_units = in_units
        self.out_units = out_units
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.last_dw = np.zeros((in_units, out_units))
        self.last_db = np.zeros((1, out_units))

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
        self.a = np.add(np.matmul(self.x, self.w), self.b)
        # print(self.a.shape)
        return self.a

    def backward(self, delta, lr=0.005, lamda = 0, momentum = True, momentum_gamma = 0.9):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # print(delta.shape)
        # print(self.w.shape)
        # print(self.x.shape)
        # print("-----------------")

        self.d_x = np.dot(self.w, delta.T)
        self.d_w = np.matmul(self.x.T, delta)
        self.d_b = np.matmul(np.ones((1,delta.shape[0])), delta)
        # print(self.d_b.shape)

        if momentum:
            if NG_IMPLEMENTATION:
                self.d_w = momentum_gamma*self.last_dw + ((1. - momentum_gamma)*self.d_w)
                self.last_dw = self.d_w
                self.d_b = momentum_gamma*self.last_db + ((1. - momentum_gamma)*self.d_b)
                self.last_db = self.d_b
            else:
                self.d_w = momentum_gamma*self.last_dw + self.d_w
                self.last_dw = self.d_w
                self.d_b = momentum_gamma*self.last_db + self.d_b
                self.last_db = self.d_b

        self.w = np.add((1.-lamda)*self.w, lr * self.d_w)
        self.b = np.add((1.-lamda)*self.b, lr * self.d_b)
        return self.d_x.T

    def deepcopy(self):
        res = Layer(self.in_units, self.out_units)

        res.w = self.w.copy()
        res.b = self.b
        if self.x is not None: res.x = self.x.copy()
        if self.a is not None: res.a = self.a.copy()
        res.in_units = self.in_units
        res.out_units = self.out_units
        if self.d_x is not None: res.d_x = self.d_x.copy()
        if self.d_w is not None: res.d_w = self.d_w.copy()
        res.d_b = self.d_b
        if self.last_dw is not None: res.last_dw = self.last_dw.copy()

        return res


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

        self.export_1 = True
        self.export_2 = True

        self.config = config

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
        targets, logits = np.atleast_2d(targets, logits)

        return -1. * np.sum([np.dot(targets[i], np.log(logits[i]+eps)) for i in range(targets.shape[0])])

    def grad_loss(self, logits, targets):
        '''
        compute the gradient w.r.t y of cross-entropy loss
        '''
        # return -1. * np.dot(1./logits, targets)
        # This should also backpropagate through Softmax as well
        return np.subtract(targets, logits)

    def backward(self, lr=0.005, lamda=0, momentum=True, momentum_gamma=0.9):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = self.grad_loss(self.y, self.targets)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr=lr, lamda=lamda, momentum=momentum, momentum_gamma=momentum_gamma)

    def deepcopy(self):
        res = Neuralnetwork(self.config)
        for i in range(len(self.layers)):
            res.layers[i] = self.layers[i].deepcopy()
        res.x = self.x.copy()
        res.y = self.y.copy()
        res.targets = self.targets.copy()

        return res


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    bs = config["batch_size"]                   # Batch Size
    lr = config["learning_rate"]                # Learning rate
    lamda = config["L2_penalty"]                # Regularization parameter
    momentum = config["momentum"]               # Momentum flag
    momentum_gamma = config["momentum_gamma"]   # Momentum param

    history = {"trloss":[],"tracc":[], "valloss":[], "valacc":[], "model":[]}
    bestmodel = None

    for e in range(config["epochs"]):
        loss_sum = 0.
        b_start = 0
        correct = 0
        print("----Epoch %d ---" % e)
        while b_start < x_train.shape[0]-1:

            # Forward pass
            b_end = min(x_train.shape[0], b_start+bs)
            logits, loss = model.forward(x_train[b_start:b_end], y_train[b_start:b_end])

            # Backwards pass
            model.backward(lr=lr, lamda=lamda, momentum=momentum, momentum_gamma=momentum_gamma)

            # Calculate loss and accuracy
            correct += np.sum(np.argmax(logits, axis=1) == np.argmax(y_train[b_start:b_start+bs, :], axis=1))
            loss_sum += loss
            b_start += bs

        print("\tTrain loss:%f, acc:%f" % (loss_sum / float(x_train.shape[0]), correct / float(x_train.shape[0])))

        # print("\tTr Acc: \t%f" % ())

        history["trloss"].append(loss_sum / float(x_train.shape[0]))
        history["tracc"].append(correct / float(x_train.shape[0]))

        valloss, valacc = test(model, x_valid, y_valid)
        print("\tValid loss:%f, acc:%f" % (valloss, valacc))
        history["valloss"].append(valloss)
        history["valacc"].append(valacc)

        history["model"].append(model.deepcopy())

        # print(len(history["valloss"][-6:]))
        if len(history["valloss"][-6:]) == 5 and (np.greater(np.diff(history["valloss"][-6:]), 0.0)).all():
            # print(np.diff(history["valloss"][-6:]))
            # print(np.greater(np.diff(history["valloss"][-6:]), 0.0))

            print("Early Stopping")
            bestmodel = history["model"][-6:][int(np.argmin(history["valloss"][-6:]))]
            break
    
    if bestmodel is None: 
        bestmodel = history["model"][-1]

    return history, bestmodel


def test(model, X_test, y_test, verbose=False):
    """
    Calculate and return the accuracy on the test set.
    """

    logits, loss = model.forward(X_test, y_test)
    correct = np.sum(np.argmax(logits, axis=1) == np.argmax(y_test, axis=1))

    loss = loss / float(x_train.shape[0])
    acc = correct / float(x_train.shape[0])

    if verbose:
        print("Test set: x:%s, y:%s"% (str(X_test.shape), str(y_test.shape)))
        print("\tLoss: \t%f" % (loss))
        print("\tAcc:\t%f" % (acc))
    return loss, acc
    

def plot_metric(trdata, valdata, title, ylabel, savename):
    plt.figure(figsize=(4,4), dpi=200)
    plt.plot(np.arange(len(trdata)), trdata)
    plt.plot(np.arange(len(valdata)), valdata)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig("./images/%s.png"%savename)

def plot_history(history, title_append, savename):
    # Loss figure
    plt.figure(dpi=200)
    plt.plot(np.arange(len(history["trloss"])), history["trloss"])
    plt.plot(np.arange(len(history["valloss"])), history["valloss"])
    plt.title("Training epochs vs. train/validation Loss for %s"%title_append)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig("./images/%s_loss.png"%savename)

    plt.figure(dpi=200)
    plt.plot(np.arange(len(history["tracc"])), history["tracc"])
    plt.plot(np.arange(len(history["valacc"])), history["valacc"])
    plt.title("Training epochs vs. train/validation Accuracy for %s"%title_append)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.savefig("./images/%s_acc.png"%savename)


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    print("--- Config settings ---")
    for k, v in config.items():
        print("%s: %s"%(str(k), str(v)))

    # Create the model
    model  = Neuralnetwork(config)

    print("--- Model summary ---")
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if isinstance(layer, Layer):
            print("Layer %d: %d inputs, %d outputs" % (i, layer.w.shape[0], layer.w.shape[1]))

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
    print("--- Training set: x: %s y: %s ---\n--- Validation set: x: %s y: %s ---" % 
          (str(x_train.shape), str(y_train.shape), str(x_valid.shape), str(y_valid.shape)))
    history, bestmodel = train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(bestmodel, x_test, y_test, verbose=True)

    # plot_metric(history["trloss"], history["valloss"], "Epoch vs Training and Validation Loss", "Loss", "3c_trloss")
    plot_history(history, "L2 penalty: 0.001", "3d_l2p001")

