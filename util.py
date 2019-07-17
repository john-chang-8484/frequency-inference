import pickle
import datetime
import math
import numpy as np

# constants
SAVE_DIR = './data'


# lookup table for the gammaln function
gammaln = np.cumsum(np.array([0.] + [math.log(i) for i in range(1, 7)]))


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print('File "%s" saved.' % filename)


def get_filepath(graphnm):
    return '%s/%s_%s.p' % (SAVE_DIR, graphnm,
        datetime.datetime.now().strftime('%S_%M_%H_%d_%m_%Y'))


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


'''
# deterministically sample n numbers (from 0 to p.size-1) with a probability distribution p
def deterministic_sample(n, p):
    return np.clip(np.floor(np.cumsum(p) * n), 0, p.size-1).astype(int)
'''

