import neuralnet
import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn import datasets
from PCA import PCA

NUM_CATEGORIES = 10


def from_one_hot(labels, num_cats):
    return np.matmul(np.arange(1, num_cats + 1), labels.T)


# Gets a sample from each category
# def get_samples(labels, num_cats):
#     samples = []
#     lbls_needed = list(range(1, num_cats + 1))
#     cat_labels = from_one_hot(labels, NUM_CATEGORIES)
#     for i in range(cat_labels.shape[0]):
#         if cat_labels[i] in lbls_needed:
#             lbls_needed.remove(cat_labels[i])
#             samples.append(i)
#         if (len(cat_labels) == 0):
#             break
#     return samples

config = neuralnet.load_config("./")
x_train, y_train = neuralnet.load_data(path="./", mode="train")
# sample_idx = get_samples(y_train, NUM_CATEGORIES)
# x_sample = x_train[sample_idx]
# y_sample = y_train[sample_idx]
x_sample = x_train[0,:].reshape(1,784)
y_sample = y_train[0].reshape(1,10)

print(x_sample.shape, y_sample.shape)

epsilon = 1e-2
layer = 0
net = neuralnet.Neuralnetwork(config)


# approximation 
i = 0
j = 0
counter = 0
layer_layers = [l for l in net.layers if isinstance(l, neuralnet.Layer)]
for l in layer_layers:
    print("new layer")
    for i in range(l.w.shape[0]):
        for j in range(l.w.shape[1]):
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
