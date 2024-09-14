import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

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

# policy = np.ones((n_states,n_actions))/n_actions
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
policy = OPT_PI

def bellman_v(R,P,policy,initial_value,gamma):
    v = np.ones(n_states)*initial_value
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
    

def bellman_q(R,P,policy,initial_value,gamma):
    q = np.ones((n_states,n_actions))*initial_value
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

gammas = [0.01, 0.5, 0.99]
for init_value in [-10, 0, 10]:
    fig, axs = plt.subplots(2, len(gammas))
    fig.suptitle(f"$V_0$: {init_value}")
    for i, gamma in enumerate(gammas):
        v_error, v = bellman_v(R,P,policy,init_value,gamma)
        r = axs[0][i].imshow(v.reshape(3,3), interpolation='nearest')
        plt.colorbar(r, ax=axs[0][i]) 
        if v_error[len(v_error)-1] == 0:
            v_error[len(v_error)-1] = 0.00001
        axs[1][i].plot(np.arange(len(v_error)), np.log(v_error))


        axs[0][i].set_title(f'$\gamma$ = {gamma}')
        axs[0][i].set_ylabel("heatmap of v(s)")
        axs[1][i].set_title(f'$\gamma$ = {gamma}, log bellman error for v(s)')
        axs[1][i].set_xlabel("time-step")
        axs[1][i].set_ylabel("log of total absolute Bellman error")

    fig, axs = plt.subplots(n_actions + 1, len(gammas))
    fig.suptitle(f"$Q_0$: {init_value}")


    for i, gamma in enumerate(gammas):
        q_error, q = bellman_q(R,P,policy,init_value,gamma)
        print(q)
        for a in range(n_actions):
            r = axs[a][i].imshow(q[:,a].reshape(3,3), interpolation='nearest')
            axs[a][i].set_ylabel(f'$action$ = {a}',rotation='horizontal')
            plt.colorbar(r, ax=axs[a][i]) 
        if q_error[len(q_error)-1] == 0:
            q_error[len(q_error)-1] = 0.00001
        axs[-1][i].plot(np.arange(len(q_error)), np.log(q_error))
        axs[0][i].set_title(f'$\gamma$ = {gamma}, heatmap for Q(s,a)')
        axs[-1][i].set_title(f'$\gamma$ = {gamma}, log bellman error for Q(s,a)')
        axs[-1][i].set_xlabel("time-step")
        axs[-1][i].set_ylabel("log of total absolute Bellman error")

    plt.show()
# v_error, v= bellman_v(R,P,policy,10, 0.99)
# q_error, q = bellman_q(R,P,policy,10, 0.99)
# print(v)
# print(q)