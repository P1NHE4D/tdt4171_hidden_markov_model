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
        fv = self.filtering(ev, prior)
        b = np.ones((ev.shape[0] + 2, prior.shape[0]))
        t = ev.shape[0]
        for i in range(t, -1, -1):
            b[i] = self.backward(b[i+1], ev[i-1], fv[i-1])
        return b

    def forward(self, fv, ev):
        p = self.sm[ev] * np.dot(self.tm, fv)
        return p / p.sum()

    def backward(self, b, ev, fv):
        a = self.sm[ev] * b
        b = np.dot(a, self.tm)
        c = fv * b
        d = c / c.sum()
        return d
