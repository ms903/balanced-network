import numpy as np
import matplotlib as mpl
import neuron as lif
import matplotlib.pyplot as plt

# some constant
W = 5  # when w = 1.58 the average of spike is around 10.1
rx = 20  # Hz
E = 100  # numbers of excitatory input
I = 100  # numbers of inhibitory input
duration = 1
dt = 1e-4
W_e = W / np.sqrt(E)
W_i = W / np.sqrt(I)



def generate_neuron_withEI(W, rx, E, I, duration, dt, W_e, W_I):
    # to generate a neuron that receives excitatory and inhibitory inputs

    x = lif.Neuron(duration, dt)
    Exci = []
    Inhi = []
    for i in range(E):
        Exci.append(lif.Neuron(duration, dt))
        Exci[i].poisson_spike(rx)
        Exci[i].Si = Exci[i].Si * W_e
    for i in range(I):
        Inhi.append(lif.Neuron(duration, dt))
        Inhi[i].poisson_spike(rx)
        Inhi[i].Si = Inhi[i].Si * W_i

    input = np.zeros(x.n_bins)
    for i in range(x.n_bins):
        for e in range(E):
            input[i] = input[i] + Exci[e].Si[i]
        for j in range(I):
            input[i] = input[i] - Inhi[j].Si[i]

    for t in range(x.n_bins - 1):
        x.recieve_once(input, t)

    return x


trials = 25
x = []
S = []
V = []

for i in range(trials):
    x.append(generate_neuron_withEI(W, rx, E, I, duration, dt, W_e, W_i))
    S.append(x[i].Si * dt)
    V.append(x[i].V)

# plt.plot(np.linspace(0,duration,int(duration/dt)),V[0])
# plt.show()

# calculate some constant and fano factor
spike = np.sum(S, axis=1)
print("mean of spikes:", spike.mean())
print("variance of spikes:", spike.var())
print("fano factor:", spike.var() / spike.mean())

# calculate the mean and variance of membrane potential
mu = np.mean(V, axis=1)
sigma_sq = np.var(V, axis=1)
print("mu of membrane is ", mu.mean())
print("the theoretical value of mu is: ", E * (W_e - W_i) * rx)
print("sigma square of membrane is ", sigma_sq.mean())
print("the theoretical value of sigma is ", E * (W_e ** 2 + W_i ** 2) * rx)

# plot the mean and variance of the membrane potential
plt.plot(mu)
plt.plot(sigma_sq)
plt.show()

# plot every neuron's spike
# for i in range(int(duration / dt)):
#     for j in range(trials):
#         if S[j][i] > 0:
#             plt.plot(i, 2 * j + 1, '|', c='r')
#
# plt.savefig("spikes.png", dpi=500)
# plt.show()

# plt.imshow(np.array(S).astype(int))
# print(S)
# print(np.array(S).shape)
# plt.show()
