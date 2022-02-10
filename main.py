import numpy as np
from hmm.utils import print_probabilities
from hmm.core import HMM


def main():
    tm = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
    ])
    sm = np.array([
        [0.1, 0.8],
        [0.9, 0.2]
    ])
    hmm = HMM(tm, sm)
    ev = np.array([1, 1, 0, 1, 1])
    prior = np.array([0.5, 0.5])
    print_probabilities(hmm.filtering(ev=ev, prior=prior))
    print_probabilities(hmm.smoothing(ev=ev, prior=prior))


if __name__ == '__main__':
    main()
