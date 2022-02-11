import numpy as np


class HMM:

    def __init__(self, tm, sm):
        """
        :param tm: Transition matrix describing the probabilities for transitioning from state i to j
        :param sm: Sensor matrix describing the probabilities for observing evidence i in state j
        """
        self.tm = tm
        self.sm = sm

    def filtering(self, ev, prior):
        """
        :param ev: observed evidence for time period t (vector<t>)
        :param prior: prior probability for each state j (vector<j>)
        :return:
        """
        t = ev.shape[0]
        fv = np.zeros((t + 1, prior.shape[0]))

        # initialise first forward value with prior probability
        fv[0] = prior
        for i in range(1, t + 1):
            fv[i] = self.forward(fv[i - 1], ev[i-1])
        return fv

    def smoothing(self, ev, prior):
        """
        :param ev: observed evidence for time period t (vector<t>)
        :param prior: prior probability for each state j (vector<j>)
        :return: backward messages for t+1 --> 1 for each state j (array<t+1, j>)
        """
        # compute forward values (fv)
        fv = self.filtering(ev, prior)[1:]

        # initialize backward values (bv) with 1s
        bv = np.ones((ev.shape[0] + 1, prior.shape[0]))

        result = np.zeros((ev.shape[0], prior.shape[0]))

        t = ev.shape[0] - 1
        for i in range(t, -1, -1):
            # compute smoothed probability for state t
            prob = fv[i] * bv[i + 1]
            result[i] = prob / prob.sum()

            # update backwards value based on succeeding backwards value
            bv[i] = self.backward(bv[i + 1], ev[i])
        return result

    def forward(self, fv, ev):
        p = self.sm[ev] * np.dot(self.tm, fv)
        return p / p.sum()

    def backward(self, b, ev):
        return np.dot(self.sm[ev] * b, self.tm)
