import neuralnet
import numpy as np
from PCA import PCA


NUM_CATEGORIES = 10


def main():

    # Build network with original config for 3 layer network
    config = neuralnet.load_config("./", fname="config_original.yaml")
    net = neuralnet.Neuralnetwork(config)

    # Load training data
    x_train, y_train = neuralnet.load_data(path="./", mode="train")
    x_sample = x_train[0,:].reshape(1,784)
    y_sample = y_train[0].reshape(1,10)
    print("x, y shapes: %s, %s" % (str(x_sample.shape), str(y_sample.shape)))

    # Set parameters
    epsilon = 1e-1

    # Get list of weight locations to test
    locations = get_experiment_weights(net)
    # locations = get_all_weights(net)      # Tests all weights in network
    for location in locations:
        if location[3] == 'w':
            check_weight(net, location, epsilon, x_sample, y_sample)
        elif location[3] == 'b':
            check_bias(net, location, epsilon, x_sample, y_sample)



def get_experiment_weights(net):
    weights_to_check = [(2, 0, 0, 'b'), # 1 output bias
                        (0, 0, 0, 'b'),  # 1 hidden bias
                        (2, 0, 0, 'w'), (2, 0, 0, 'w'), # 2 hidden-output weights
                        (0, 0, 0, 'w'), (0, 0, 0, 'w'), # 2 input-hidden weights 
                        ]
    return weights_to_check


def check_weight(net, location, epsilon, x_sample, y_sample):
    # Gets E(w+e)
    layer = location[0]
    i = location[1]
    j = location[2]

    print("Testing Weight in Layer %d, for layer input %d, layer output %d..." %(layer, i, j))

    approx = compute_approx(net, layer, i, j, epsilon, x_sample, y_sample)

    actual = compute_actual(net, layer, i, j, epsilon, x_sample, y_sample)

    compare_gradients(approx, actual, epsilon)


def check_bias(net, location, epsilon, x_sample, y_sample):
    # Gets E(w+e)
    layer = location[0]
    i = location[1]
    j = location[2]

    print("Testing Bias in Layer %d, for layer output %d..." %(layer, j))

    approx = compute_approx(net, layer, i, j, epsilon, x_sample, y_sample, bias=True)

    actual = compute_actual(net, layer, i, j, epsilon, x_sample, y_sample, bias=True)

    compare_gradients(approx, actual, epsilon)


def compare_gradients(approx, actual, eps):
    """ Tests gradients from approximation and actual test """
    diff = np.abs(approx - actual)
    pass_test = (diff <= eps**2)
    if not pass_test:
        print("\tFAILED")
        print("\t\tApproximation: %f, Actual: %f" % (actual, approx))
        print("\t\t%f - %f = %f > %f" % (actual, approx, diff, eps**2))
    else:
        print("\tPASSED")


def compute_approx(net, layer, i, j, eps, x_sample, y_sample, bias=False):
    """
    Computes actual gradient by running example through network
    with a single weight modulated by eps, and computing the 
    approximation of the 1st derivative
    """
    # Gets E(w+e)
    if not bias:
        net.layers[layer].w[i, j]  += eps
    else:
        net.layers[layer].b[i, j]  += eps
    y_hi, loss_hi = net.forward(x_sample, y_sample)

    # Gets E(w-e)
    net.layers[layer].w[i, j]  -= (2*eps)
    if not bias:
        net.layers[layer].w[i, j]  -= (2*eps)
    else:
        net.layers[layer].b[i, j]  -= (2*eps)
    y_lo, loss_lo = net.forward(x_sample, y_sample)

    # Sets weight back to original for actual gradient computation

    if not bias:
        net.layers[layer].w[i, j]  += eps
    else:
        net.layers[layer].b[i, j]  += eps

    # Calculates approximation
    approx = (loss_hi - loss_lo)/(2*eps)

    return approx


def compute_actual(net, layer, i, j, eps, x_sample, y_sample, bias=False):
    """Computes actual gradient by running example through network"""
    
    _, _ = net.forward(x_sample, y_sample)

    net.backward()

    #because the definition is different
    if not bias:
        actual = (- net.layers[layer].d_w[i,j])
    else:
        actual = (- net.layers[layer].d_b[i,j])

    return actual


def from_one_hot(labels, num_cats):
    """ Converts labels to one hot encodings """
    return np.matmul(np.arange(1, num_cats + 1), labels.T)


def get_all_weights(net):
    """
    Internal testing method
    """
    weights_to_check = []
    layer_idxs = [l for l in range(len(net.layers)) if isinstance(net.layers[l], neuralnet.Layer)]
    for l in layer_idxs:
        print("new layer")
        lay = net.layers[l]
        for i in range(lay.w.shape[0]):
            for j in range(lay.w.shape[1]):
                weights_to_check.append( (l, i, j, 'w') )
    return weights_to_check


if __name__ == "__main__":
    main()
