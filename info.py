import numpy as np
from sys import argv
from util import load_data

MAXSTRLEN = 100

def main():
    data = load_data(argv[1])
    for key in data:
        s = str(data[key])
        if len(s) < MAXSTRLEN or key in ['get_v1', 'get_omega_list', 'get_estimator']:
            print(key, ':', s)
    if len(argv) > 2:
        for key in data:
            print(key, ':', repr(data[key]))

if __name__ == '__main__':
    main()

