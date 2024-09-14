import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

OPT_PI = np.zeros((n_states, n_actions))
OPT_PI[0,1] = 1
OPT_PI[3,1] = 1

OPT_PI[4,2] = 1
OPT_PI[6,2] = 1
OPT_PI[1,2] = 1
OPT_PI[7,2] = 1

OPT_PI[5,3] = 1
OPT_PI[8,3] = 1

OPT_PI[2,4] = 1
policy_random = np.ones((n_states, n_actions))/n_actions
pi_opt = OPT_PI

R = np.zeros((n_states + 1, n_actions))
P = np.zeros((n_states + 1, n_actions, n_states + 1))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        if terminated:
            P[s, a, n_states] = 1.0
        else:
            P[s, a, s_next] = 1.0


def policy_improvement(v,gamma):
    new_policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_s_a = np.zeros((n_actions))
        for a in range(n_actions):
            for s_next in range(n_states+1):
                if s_next == n_states:
                    q_s_a[a] += P[s,a,s_next]*R[s,a]
                else:
                    q_s_a[a] += P[s,a,s_next]*(R[s,a]+gamma*v[s_next])
        max_q = np.max(q_s_a)
        max_a, = np.where(q_s_a==max_q)
        for index in max_a:
            new_policy[s,index] = 1/len(max_a)
    return new_policy

def policy_evaluation(policy,init_v,gamma):
    v = init_v
    error = 10000
    v_error = []
    while error > 0.00001 :
        error = 0
        bellman_error = 0
        for s in range(n_states):
            v_c = 0
            for a in range(n_actions):
                for s_next in range(n_states+1):
                    if s_next == n_states:
                        v_c += policy[s,a]*P[s,a,s_next]*R[s,a]
                    else:
                        v_c += policy[s,a]*P[s,a,s_next]*(R[s,a]+gamma*v[s_next])
            error = max(error, np.abs(v[s]-v_c))
            bellman_error += np.abs(v[s]-v_c)
            v[s] = v_c
        v_error.append(bellman_error)
    return v_error, v

def policy_iteration(policy,initial_value,gamma):
    policy_stable = False
    bellman_error = []
    v = np.ones(n_states)*initial_value
    while not policy_stable:
        v_error, v = policy_evaluation(policy,v,gamma)
        new_policy = policy_improvement(v,gamma)
        policy_stable = np.array_equal(policy,new_policy)
        policy = new_policy
        bellman_error = bellman_error + v_error
    tot_iter = len(bellman_error)
    return policy, tot_iter, bellman_error

def generalized_policy_iteration(policy,initial_value,gamma):
    error = 10000
    v_error = []
    v = np.ones(n_states)*initial_value
    while error > 0.0001:
        error = 0
        for i in range(5):
            bellman_error = 0
            for s in range(n_states):
                v_c = 0
                for a in range(n_actions):
                    for s_next in range(n_states+1):
                        if s_next == n_states:
                            v_c += policy[s,a]*P[s,a,s_next]*R[s,a]
                        else:
                            v_c += policy[s,a]*P[s,a,s_next]*(R[s,a]+gamma*v[s_next])
                error = max(error, np.abs(v[s]-v_c))
                bellman_error += np.abs(v[s]-v_c)
                v[s] = v_c
            v_error.append(bellman_error)
        policy = policy_improvement(v,gamma)
    return policy, len(v_error), v_error

def value_iteration(initial_value,gamma):
    error = 10000
    bellman_error = []
    v = np.ones(n_states)*initial_value
    #_, v = policy_evaluation(policy,v,gamma)
    while error>0.0001:
        per_error = 0
        error = 0
        for s in range(n_states):
            v_c = 0
            q_s_a = np.zeros((n_actions))
            for a in range(n_actions):
                for s_next in range(n_states+1):
                    if s_next == n_states:
                        q_s_a[a] += P[s,a,s_next]*R[s,a]
                    else:
                        q_s_a[a] += P[s,a,s_next]*(R[s,a]+gamma*v[s_next])
            v_c = np.max(q_s_a)
            error = max(error, np.abs(v[s]-v_c))
            per_error += np.abs(v[s]-v_c)
            v[s] = v_c
        bellman_error.append(per_error)
    opt_policy = policy_improvement(v,gamma)

    return opt_policy, len(bellman_error), bellman_error


# pi, tot_iter, be = policy_iteration(policy_random,0,0.99)
# # print(pi)
# # print(pi_opt)
# assert np.allclose(pi, pi_opt)

# v = np.ones(n_states)*0
# _, v = policy_evaluation(pi_opt,v,0.99)
# pi, tot_iter, be = value_iteration(policy_random,0,0.99)
# # print(pi)
# # print(pi_opt)
# assert np.allclose(pi, pi_opt)

# pi, tot_iter, be = generalized_policy_iteration(policy_random,0,0.99)
# # print(pi)
# # print(pi_opt)
# assert np.allclose(pi, pi_opt)

fig, axs = plt.subplots(3, 7)
tot_iter_table = np.zeros((3, 7))
for i, init_value in enumerate([-100, -10, -5, 0, 5, 10, 100]):
    axs[0][i].set_title(f'$V_0$ = {init_value}')

    pi, tot_iter, be = value_iteration(init_value,0.99)
    tot_iter_table[0, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[0][i].plot(np.arange(tot_iter),be)

    pi, tot_iter, be = policy_iteration(policy_random,init_value,0.99)
    tot_iter_table[1, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[1][i].plot(np.arange(tot_iter),be)

    pi, tot_iter, be = generalized_policy_iteration(policy_random,init_value,0.99)
    tot_iter_table[2, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[2][i].plot(np.arange(tot_iter),be)

    if i == 0:
        axs[0][i].set_ylabel("VI")
        axs[1][i].set_ylabel("PI")
        axs[2][i].set_ylabel("GPI")

plt.show()

print(tot_iter_table.mean(-1))
print(tot_iter_table.std(-1))