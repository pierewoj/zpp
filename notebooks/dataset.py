import numpy as np
from scipy import misc

class Dataset():
    def __init__(self, path, batch):
        self.train = np.load(path+"_train.npy"), np.load(path+"_trainL.npy")
        self.test = np.load(path+"_test.npy"), np.load(path+"_testL.npy")
        self.valid = np.load(path+"_valid.npy"), np.load(path+"_validL.npy")
        self.train_next = 0
        self.batch = batch
        self.starting = self.train
        np.random.seed()
    
    def next_batch(self):
        assert self.train_next <= len(self.train[0])
        if self.train_next == len(self.train[0]):
            self.train_next = 0
        if self.train_next + self.batch > len(self.train[0]):
            batch =  self.train[0][self.train_next:], self.train[1][self.train_next:]
        else:
            batch = self.train[0][self.train_next:self.train_next + self.batch], self.train[1][self.train_next:self.train_next + self.batch]
        self.train_next += len(batch[0])
        return batch
    
    def shuffle(self):
        self.train_next = 0
        perm = np.random.permutation(len(self.train[0]))
        self.train = self.train[0][perm], self.train[1][perm]
    
    def reset(self):
        self.train = self.starting
        np.random.seed(self.seed)
        
        
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]