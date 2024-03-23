import numpy as np
import neuron as lif


class network:

    def __init__(self, num, rx, duration, dt, tau, V_th, V_set, V_reset, J_EE, J_EI, J_IE, J_II, J_XE, J_XI):
        # num -> numbers of neurons in the network
        self.num = num
        self.duration = duration
        self.dt = dt
        self.tau = tau
        self.rx = rx

        # the constant of weight
        self.weigh_mat = np.zeros([4, 4])
        self.weigh_mat[1][1] = J_EE
        self.weigh_mat[1][2] = J_EI
        self.weigh_mat[2][1] = J_IE
        self.weigh_mat[2][2] = J_II
        self.weigh_mat[3][1] = J_XE
        self.weigh_mat[3][2] = J_XI

        # initialize each numbers of each type of neurons
        self.E = 0
        self.I = 0
        self.X = 0

        # initialize the type of each neuron
        # '1' is excitatory
        # '2' is inhibitory
        # '3' is external
        self.type = np.zeros(num)

        # initialize the neurons and put into list 'x'
        # each neuron is numbered from 0~num-1
        self.neurons = []
        for i in range(num):
            self.neurons.append(lif.Neuron(duration, dt, tau, V_th, V_set, V_reset))

        # type_cnt[i][j] the number of type j that pre-synaptic to neuron i
        self.type_cnt = [[0, 0, 0, 0] for i in range(num)]

        # initialize the list that record the connection
        # if neuron j is pre-synaptic to neuron i
        # then self.connection[i][j] = 1
        self.connection = [[] for i in range(num)]

        self.all_input = [[0 for j in range(int(duration / dt) - 1)] for i in range(num)]

    def set_type(self, i, type_i):
        # set neuron i to type_i
        self.type[i] = type_i
        if type_i == 1:
            self.E += 1
        elif type_i == 2:
            self.I += 1
        elif type_i == 3:
            self.X += 1
            self.neurons[i].poisson_spike(self.rx)
        else:
            return 'type input error'

    def connect(self, pre, to):
        # make a connection from neuron 'pre' to neuron 'to'
        self.connection[to].append(pre)
        self.type_cnt[to][int(self.type[pre])] += 1

    def cal_input(self, i, t):
        # calculate the input of neuron i at time t

        # numbers of input that neuron i receives
        K = len(self.connection[i])
        if K == 0:
            return 0

        type_i = int(self.type[i])
        rst = [0., 0., 0., 0.]

        for j in range(K):
            pre = self.connection[i][j]
            type_pre = int(self.type[pre])
            weight = self.weigh_mat[type_pre][type_i] * self.neurons[pre].output[t-1]
            rst[type_pre] = rst[type_pre] + weight

        rst_w = 0
        for j in range(1, 4):
            if self.type_cnt[i][j] != 0:
                rst_w += rst[j] / np.sqrt(self.type_cnt[i][j])
        self.all_input[i][t] = rst_w
        return rst_w

    def integrate_once(self, t):
        # integrate each neuron in the network once at time t
        for i in range(self.num):
            if self.type[i] == 3:
                # if the neuron is external, skip the loop
                continue
            else:
                input_t = self.cal_input(i, t - 1)
                self.neurons[i].integrate_once(input_t, t - 1)

    def run_duration(self):
        # integrate the all duration of all neurons
        for t in range(int(self.duration / self.dt) - 1):
            self.integrate_once(t)

    def get_V_i(self, i):
        return self.neurons[i].V

    def get_V_E_neurons(self):
        V = []
        for i in range(self.num):
            if self.type[i] == 1:
                V.append(self.neurons[i].get_V())
        return np.array(V)

    def get_V_I_neurons(self):
        V = []
        for i in range(self.num):
            if self.type[i] == 2:
                V.append(self.neurons[i].get_V())
        return np.array(V)

    def get_connection(self):
        return self.connection

    def get_output_i(self, i):
        return self.neurons[i].output

    def get_output(self):
        output = []
        for i in range(self.num):
            output.append(self.neurons[i].output)
        return output

    def get_input(self):
        return self.all_input

    def get_input_i(self, i):
        return self.all_input[i]
