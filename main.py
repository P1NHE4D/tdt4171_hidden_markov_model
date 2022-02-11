import numpy as np
from hmm.utils import print_probabilities
from hmm.core import HMM


def main():
    # transition model
    tm = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
    ])

    # sensor model
    sm = np.array([
        [0.1, 0.8],
        [0.9, 0.2]
    ])

    # evidence - 1 == True; 0 == False
    ev = np.array([1, 1, 0, 1, 1])

    # prior probability
    prior = np.array([0.5, 0.5])

    # create hidden markov model
    hmm = HMM(tm, sm)

    print("Forward values:")
    print_probabilities(hmm.filtering(ev=ev, prior=prior))

    print("Backward values:")
    print_probabilities(hmm.smoothing(ev=ev, prior=prior), start_index=1)


if __name__ == '__main__':
    main()
