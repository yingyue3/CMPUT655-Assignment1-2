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


def policy_improvement(q):
    new_policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_s_a = q[s,:]
        max_q = np.max(q_s_a)
        max_a, = np.where(q_s_a==max_q)
        for index in max_a:
            new_policy[s,index] = 1/len(max_a)
    return new_policy

def policy_evaluation(policy,init_q,gamma):
    q = init_q
    error = 10000
    q_error = []
    while error > 0.00001 :
        error = 0
        bellman_error = 0
        for s in range(n_states):
            for a in range(n_actions):
                q_c = 0
                for s_next in range(n_states+1):
                    if s_next == n_states:
                        q_c += P[s,a,s_next]*R[s,a]
                    else:
                        q_next = 0
                        for a_next in range(n_actions):
                            q_next += policy[s_next,a_next]*q[s_next,a_next]
                        q_c += P[s,a,s_next]*(R[s,a]+gamma*q_next)
                error = max(error, np.abs(q[s,a]-q_c))
                bellman_error += np.abs(q[s,a]-q_c)
                q[s,a] = q_c
        q_error.append(bellman_error)
    return q_error, q

def policy_iteration(policy,initial_value,gamma):
    policy_stable = False
    bellman_error = []
    q = np.ones((n_states,n_actions))*initial_value
    while not policy_stable:
        q_error, q = policy_evaluation(policy,q,gamma)
        new_policy = policy_improvement(q)
        policy_stable = np.array_equal(policy,new_policy)
        policy = new_policy
        bellman_error = bellman_error + q_error
    tot_iter = len(bellman_error)
    return policy, tot_iter, bellman_error

def generalized_policy_iteration(policy,initial_value,gamma):
    error = 10000
    q_error = []
    q = np.ones((n_states,n_actions))*initial_value
    while error > 0.0001:
        error = 0
        for i in range(5):
            bellman_error = 0
            for s in range(n_states):
                for a in range(n_actions):
                    q_c = 0
                    for s_next in range(n_states+1):
                        if s_next == n_states:
                            q_c += P[s,a,s_next]*R[s,a]
                        else:
                            q_next = 0
                            for a_next in range(n_actions):
                                q_next += policy[s_next,a_next]*q[s_next,a_next]
                            q_c += P[s,a,s_next]*(R[s,a]+gamma*q_next)
                    error = max(error, np.abs(q[s,a]-q_c))
                    bellman_error += np.abs(q[s,a]-q_c)
                    q[s,a] = q_c
            q_error.append(bellman_error)
        policy = policy_improvement(q)
    return policy, len(q_error), q_error

def value_iteration(initial_value,gamma):
    error = 10000
    bellman_error = []
    q = np.ones((n_states,n_actions))*initial_value
    while error>0.0001:
        per_error = 0
        error = 0
        for s in range(n_states):
            for a in range(n_actions):
                q_s_a_next = 0
                for s_next in range(n_states+1):
                    if s_next == n_states:
                        q_s_a_next += P[s,a,s_next]*R[s,a]
                    else:
                        q_s_a_next += P[s,a,s_next]*(R[s,a]+gamma*np.max(q[s_next,:]))
                error = max(error, np.abs(q[s,a]-q_s_a_next))
                per_error += np.abs(q[s,a]-q_s_a_next)
                q[s,a] = q_s_a_next
        bellman_error.append(per_error)
    opt_policy = policy_improvement(q)

    return opt_policy, len(bellman_error), bellman_error


# pi, tot_iter, be = policy_iteration(policy_random,0,0.99)
# # print(pi)
# # print(pi_opt)
# assert np.allclose(pi, pi_opt)

# q = np.ones((n_states,n_actions))*0
# _, q = policy_evaluation(pi_opt,q,0.99)
# pi, tot_iter, be = value_iteration(0,0.99)
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
    axs[0][i].set_title(f'$Q_0$ = {init_value}')

    pi, tot_iter, be = value_iteration(init_value,0.99)
    tot_iter_table[0, i] = np.log(tot_iter)
    assert np.allclose(pi, pi_opt)
    axs[0][i].plot(np.arange(tot_iter),be)

    pi, tot_iter, be = policy_iteration(policy_random,init_value,0.99)
    tot_iter_table[1, i] = np.log(tot_iter)
    assert np.allclose(pi, pi_opt)
    axs[1][i].plot(np.arange(tot_iter),be)

    pi, tot_iter, be = generalized_policy_iteration(policy_random,init_value,0.99)
    tot_iter_table[2, i] = np.log(tot_iter)
    assert np.allclose(pi, pi_opt)
    axs[2][i].plot(np.arange(tot_iter),be)

    if i == 0:
        axs[0][i].set_ylabel("Bellam error for VI")
        axs[1][i].set_ylabel("Bellam error for PI")
        axs[2][i].set_ylabel("Bellam error for GPI")

    if i == 3:
        axs[2][i].set_xlabel("Total policy evaluation iteration")

plt.show()

print(tot_iter_table.mean(-1))
print(tot_iter_table.std(-1))