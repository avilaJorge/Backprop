import neuralnet
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets

NUM_CATEGORIES = 10


def from_one_hot(labels, num_cats):
    return np.matmul(np.arange(1, num_cats + 1), labels.T)


# Gets a sample from each category
def get_samples(labels, num_cats):
    samples = []
    lbls_needed = list(range(1, num_cats + 1))
    cat_labels = from_one_hot(labels, NUM_CATEGORIES)
    for i in range(cat_labels.shape[0]):
        if cat_labels[i] in lbls_needed:
            lbls_needed.remove(cat_labels[i])
            samples.append(i)
        if (len(cat_labels) == 0):
            break
    return samples

F = np.arange(1, 6)
G = 1/F
print(G)


config = neuralnet.load_config("./")
x_train, y_train = neuralnet.load_data(path="./", mode="train")
sample_idx = get_samples(y_train, NUM_CATEGORIES)
x_sample = x_train[sample_idx]
y_sample = y_train[sample_idx]

epsilon = 10e-2
net = neuralnet.Neuralnetwork(config)
Y, loss = net.forward(x_sample, y_sample)
net.backward()
print(net.layers)
print(Y)
print(np.sum(Y))
print(np.max(Y))
print(y_sample[0])

# Example usage for Sklearn StratifiedKFold
# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits()
# for train_index, test_index in skf.split(X=x_train, y=from_one_hot(y_train, NUM_CATEGORIES)):
#     print("TRAIN: ", train_index, " TEST: ", test_index)
#     A = y_train[train_index]
#     B = y_train[test_index]
#     print("Train Counts: ", np.sum(A, axis=0))
#     print("Test Counts: ", np.sum(B, axis=0))
