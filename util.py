import pickle
import datetime

SAVE_DIR = './data'

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
class TooManyIterationsError(Exception):
    def __init__(self):
        pass
'''

