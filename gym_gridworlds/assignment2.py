import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

...

def policy_improvement(**kargs):
    return

def policy_evaluation(**kargs):
    return

def policy_iteration(**kargs):
    return

def generalized_policy_iteration(**kargs):
    return

def value_iteration(**kargs):
    return

fig, axs = plt.subplots(3, 7)
tot_iter_table = np.zeros((3, 7))
for i, init_value in enumerate([-100, -10, -5, 0, 5, 10, 100]):
    axs[0][i].set_title(f'$V_0$ = {init_value}')

    pi, tot_iter, be = value_iteration(...)
    tot_iter_table[0, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[0][i].plot(...)

    pi, tot_iter, be = policy_iteration(...)
    tot_iter_table[1, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[1][i].plot(...)

    pi, tot_iter, be = generalized_policy_iteration(...)
    tot_iter_table[2, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[2][i].plot(...)

    if i == 0:
        axs[0][i].set_ylabel("VI")
        axs[1][i].set_ylabel("PI")
        axs[2][i].set_ylabel("GPI")

plt.show()

print(tot_iter_table.mean(-1))
print(tot_iter_table.std(-1))