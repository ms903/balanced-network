import numpy as np


class Neuron():
    """Initialize a Neuron"""

    def __init__(self, duration, dt, tau, V_th, V_set, V_reset):
        self.duration = duration
        self.tau = tau
        self.V_th = V_th
        self.V_set = V_set
        self.V_reset = V_reset
        self.dt = dt
        self.n_bins = int(duration / dt) - 1
        self.output = np.zeros(self.n_bins)
        self.V = np.zeros(self.n_bins)
        # V[t] is the membrane potential of the neuron at time t

    '''Poisson_spike method can output poisson spike trains'''

    def poisson_spike(self, rx):
        # make up a random spike train
        self.output = np.random.binomial(1, rx * self.dt, size=self.n_bins) / self.dt
        return self.output

    '''Euler integrate'''

    def integrate_once(self, input_t, t):
        """ input_t is the sum of input at time t, it needs to be calculated previously,
            and the pass it to this function"""
        if self.V[t] == self.V_set:
            self.output[t] = 1 / self.dt
            self.V[t + 1] = self.V_reset
        else:
            self.V[t + 1] = self.V[t] + self.dt * (-self.V[t] / self.tau + input_t)
        if self.V[t + 1] >= self.V_th:
            self.V[t + 1] = self.V_set

    '''Action, no return'''

    def receive_once(self, input_t, t):
        self.integrate_once(input_t, t)

    def get_V(self):
        return self.V

    def get_output(self):
        return self.output
