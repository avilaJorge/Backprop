class KFold(object):
    """docstring for KFold"""
    def __init__(self,):
        super(KFold, self).__init__()

    def get_split_idxs(self, num_samples, test_val_perc=0.3):
        idxs = np.arange(num_samples)

        numpy.random.shuffle(idxs)

        # train_end = num_samples * (1.-(2.*test_val_perc))

        val_end = num_samples * (1.-(test_val_perc))

        training, val = idxs[:val_end,:], idxs[val_end:,:]
        return training, val

        