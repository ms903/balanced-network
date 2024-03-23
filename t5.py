import numpy as np
import networks as bn
import matplotlib.pyplot as plt

rx = 10
tau = 20e-3
dt = 1e-4
duration = 2

E = 100
I = 100
V_th = 1
V_set = 20
V_reset = 0

w = 1.5
J_EE = w
J_IE = -w
J_II = 0
J_EI = 0
J_XI = 0
J_XE = 0

ts = np.linspace(0, duration, int(duration / dt) - 1)


def do_once():
    nw = bn.network(E + I + 1, rx, duration, dt, tau, V_th, V_set, V_reset, J_EE, J_EI, J_IE, J_II, J_XE, J_XI)

    for i in range(E):
        nw.set_type(i, 1)
        nw.connect(i, E + I)
        nw.neurons[i].poisson_spike(rx)

    for i in range(E, E + I):
        nw.set_type(i, 2)
        nw.connect(i, E + I)
        nw.neurons[i].poisson_spike(rx)

    nw.set_type(E + I, 1)
    nw.run_duration()

    return nw


# x = do_once()

# plt.plot(x.get_E_neuron_V()[E])
# plt.show()
# plt.plot(x.get_input(E+I))
# plt.show()

trials = 15
V = []
S = []
mu = []
sigma = []
for i in range(trials):
    V.append(do_once())
    print(i)
    mu.append(np.mean(V[i].get_V_i(E + I)[1000:-1]))
    sigma.append(np.var(V[i].get_V_i(E + I)[1000:-1]))
    S.append(V[i].neurons[E + I].output * dt)
    # plt.plot(ts,V[i].get_E_neuron_V()[E])
    # plt.show()
    # plt.plot(ts,V[i].neurons[E+I].output * dt)
    # plt.show()
    # plt.plot(ts,V[0].get_input()[E+I])

# plt.show()

plt.plot(mu, c='r')
plt.plot(sigma, c='b')
plt.show()
print(np.mean(mu))
print(np.mean(sigma))

# for i in range(int(duration / dt) - 1):
#     for j in range(trials):
#         if S[j][i] > 0:
#             plt.plot(i, 2 * j + 1, '|', c='r')
#
# plt.savefig("spikes2.png", dpi=500)
# plt.show()

spike = np.sum(S, axis=1)
print(spike)
print("mean spike per sec ", np.mean(spike))

# plt.plot(do_once())
# plt.show()
