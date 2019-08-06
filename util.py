import pickle
import datetime
import math, random, string
import numpy as np
from qinfer import Distribution


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


def randstr(n):
    """ create a random string for file disambguation of files created
        within 1s of each other. """
    return ''.join(
        random.SystemRandom().choice(string.ascii_uppercase)
        for i in range(n) )

def get_filepath(graphnm):
    return '%s/%s_%s_%s.p' % (SAVE_DIR, graphnm, randstr(3),
        datetime.datetime.now().strftime('%S_%M_%H_%d_%m_%Y'))


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_numeric_class_vars(a_class):
    return {
        key: vars(a_class)[key]
            for key in vars(a_class)
            if type(vars(a_class)[key]) in [int, float]
    }




# deterministically sample n numbers (from 0 to p.size-1) with a probability distribution p
def deterministic_sample(n, p):
    cdf = np.cumsum(p)
    comparisons = np.linspace(1/(2*n), 1-1/(2*n), n)
    ans = np.zeros(n, dtype=np.int64)
    i = 0
    for j in range(0, n):
        while cdf[i] < comparisons[j]:
            i += 1
        ans[j] = i
    return ans


# given two strings, find the substring that differs between them,
#   return what this substring is for st1
#   pass [] as empty to perform this operation on lists instead
def diff(st1, st2, empty=''):
    if len(st1) == 0:
        return empty
    if len(st2) == 0:
        return st1
    if st1[0] == st2[0]:
        return diff(st1[1:], st2[1:], empty)
    if st1[-1] == st2[-1]:
        return diff(st1[:-1], st2[:-1], empty)
    return st1


# compute the Gini coefficient for a distribution
# https://en.wikipedia.org/wiki/Gini_coefficient#Discrete_probability_distribution
def gini(dist):
    S = np.cumsum(np.sort(dist, axis=None))
    return 1. - (1 + 2. * np.sum(S[0:-1]) / S[-1]) / dist.size


def idprn(x):
    print(x)
    return x



