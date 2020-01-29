import unittest
import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn
import neuralnet as nnt
from torch.autograd import Variable


class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)
        self.model = None
        self.config = None
        self.setup()

    def setup(self):
        # Load the configuration.
        self.config = nnt.load_config("./")
        # Create the model
        self.model = nnt.Neuralnetwork(self.config)

    def teardown(self):
        print("___________Teardown____________")

    def from_one_hot(self, labels, num_cats):
        return np.matmul(np.arange(0, num_cats), labels.T)

    def test_something(self):


        Z = np.load("z.npy")
        T = np.load("targets.npy")

        T_conv = self.from_one_hot(T, 10)

        output = Variable(torch.from_numpy(Z).float())
        target = Variable(torch.from_numpy(T_conv).long())

        criterion = nn.CrossEntropyLoss()
        loss_correct = criterion(output, target)

        Z = nnt.softmax(Z)

        loss_actual = self.model.loss(Z, T)
        npt.assert_almost_equal(loss_actual, loss_correct.item() * Z.shape[0], decimal=2)

    def test_simply_xent(self):

        output = Variable(torch.FloatTensor([0, 0, 0, 1])).view(1, -1)
        target = Variable(torch.LongTensor([3]))

        z =  np.array([0.1749,0.1749,0.1749,0.4754]).reshape((1,4))
        t = np.array([0, 0, 0, 1]).reshape((1,4))

        criterion = nn.CrossEntropyLoss()
        loss_correct = criterion(output, target)
        loss_actual = self.model.loss(z, t)

        npt.assert_almost_equal(loss_actual, loss_correct.item(), decimal=2)




if __name__ == '__main__':
    unittest.main()
