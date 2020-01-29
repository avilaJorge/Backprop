import neuralnet
import numpy as np
from PCA import PCA

NUM_CATEGORIES = 10


def from_one_hot(labels, num_cats):
    return np.matmul(np.arange(1, num_cats + 1), labels.T)

def get_all_weights(net):
    weights_to_check = []
    layer_idxs = [l for l in range(len(net.layers)) if isinstance(net.layers[l], neuralnet.Layer)]
    for l in layer_idxs:
        print("new layer")
        lay = net.layers[l]
        for i in range(lay.w.shape[0]):
            for j in range(lay.w.shape[1]):
                weights_to_check.append( (l, i, j) )
    return weights_to_check


config = neuralnet.load_config("./", fname="config_original.yaml")
x_train, y_train = neuralnet.load_data(path="./", mode="train")
x_sample = x_train[0,:].reshape(1,784)
y_sample = y_train[0].reshape(1,10)

print("x, y shapes: %s, %s" % (str(x_sample.shape), str(y_sample.shape)))

epsilon = 1e-2
layer = 0
net = neuralnet.Neuralnetwork(config)

weights_to_check = [(0), # 1 output bias
                    (),  # 1 hidden bias
                    (), (), # 2 hidden-output weights
                    (), (), # 2 input-hidden weights 
                    ]


# approximation 
counter = 0


            # Gets E(w+e)
            net.layers[layer].w[i, j] += epsilon
            y_hi, loss_hi = net.forward(x_sample, y_sample)

            # Gets E(w-e)
            net.layers[layer].w[i, j]  -= (2*epsilon)
            y_lo, loss_lo = net.forward(x_sample, y_sample)

            # Calculate approx
            approx = (loss_hi - loss_lo)/(2*epsilon)


            #Sets w back to original
            net.layers[layer].w[i, j]  += epsilon

            # Forward for safety, back, get weight gradient
            _, _ = net.forward(x_sample, y_sample)
            net.backward()
            #because the definition is different
            actual = - net.layers[layer].d_w[i,j]

            # Compare
            diff = np.abs(approx - actual)
            pass_test = (diff <= epsilon**2)
            if not pass_test:
                print(diff, actual, approx)
            # print(approx, actual)
            # assert(pass_test)
            counter += 1
print(counter)


# Example usage for Sklearn StratifiedKFold
# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits()
# for train_index, test_index in skf.split(X=x_train, y=from_one_hot(y_train, NUM_CATEGORIES)):
#     print("TRAIN: ", train_index, " TEST: ", test_index)
#     A = y_train[train_index]
#     B = y_train[test_index]
#     print("Train Counts: ", np.sum(A, axis=0))
#     print("Test Counts: ", np.sum(B, axis=0))
