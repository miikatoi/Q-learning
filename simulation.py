# Reinforcement Learning
# Miika Toikkanen

'''
The environment is represented as matrix of rewards with dimension (state, action).
Similarly Q-values are stored with matrix of dimension (state, action).
Actions are integers. 0 for left, 1 for right.
State is an integer from 0 to 6, encoding 7 states: first,a,b,c,d,e,final
'''

import numpy as np
import matplotlib.pyplot as plt
import random

class Environment():
    '''deterministic walk environment'''
    def __init__(self):
        '''Define reward matrix index as (states, actions)'''
        self.rewards = np.zeros((7,2))
        self.rewards[5,1] = 4
        self.rewards[1,0] = 1

    def step(self, s, a):
        '''take one step
        returns reward, the next state and whether it is terminal state'''
        reward = self.rewards[s, a]
        if a == 1:
            next_state = s + 1
        else:
            next_state = s - 1 
        return reward, next_state, self.isterminal(next_state)
    
    def isterminal(self, state):
        '''return true if state is terminal (index 0 or 6)'''
        return state in [0,6]

class Agent():
    '''Q-learning agent'''
    def __init__(self):
        # Init Q values to zero
        self.Q = np.zeros((7,2))

    def initialize_S(self):
        '''Set initial state as c (index 3)'''
        self.s = 3

    def sample_Q(self, epsilon):
        '''epsilon greedy exploration. 
        returns state-action pair'''
        if random.uniform(0, 1) > epsilon:
            #explore
            a = random.choice([0, 1])
        else:
            #exploit
            a = np.argmax(self.Q[self.s])
        return self.s, a

    def update_s(self, s):
        '''update current state'''
        self.s = s

    def update_Q(self, s, a, r, s_, gamma, alpha):
        '''update the Q-values'''
        self.Q[s, a] += alpha * (r + gamma * self.Q[s_].max() - self.Q[s, a])

    def result(self):
        '''return the optimized Q-values'''
        return self.Q

def Q_learning(episodes, epsilon, gamma, alpha):

    # init environment and agent
    agent = Agent()
    env = Environment()

    for _ in range(episodes):
        # initial state
        agent.initialize_S()
        while True:
            # choose action from Q
            s, a = agent.sample_Q(epsilon)
            # take action, observe reward, next state and whether it is terminal
            r, s_, isterminal = env.step(s, a)
            # update Q-values
            agent.update_Q(s, a, r, s_, gamma, alpha)
            # update current state
            agent.update_s(s_)
            # stop if in terminal state
            if isterminal:
                break

    # return optimized Q
    return agent.result()

if __name__=='__main__':

    # fix seed
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    episodes = 10000
    alpha = 0.1

    # i discount high (gamma is high), exploitation high (1 - epsilon is high)
    case1 = Q_learning(
        episodes = episodes, 
        epsilon = 0.3, 
        gamma = 0.7,
        alpha = alpha)

    # ii discount low (gamma is low), exploitation low (1 - epsilon is low)
    case2 = Q_learning(
        episodes = episodes,
        epsilon = 0.7,
        gamma = 0.3,
        alpha = alpha)

    print('\ni discount high, exploitation high')
    print('Q-values:\n', case1)
    print('policy:\n', np.argmax(case1, axis=1)[1:-1])
    print('\nii discount low, exploitation low')
    print('Q-values:\n', case2)
    print('policy:\n', np.argmax(case2, axis=1)[1:-1])

    width = 0.35
    x = np.arange(7)
    fig, ax = plt.subplots()
    ax.set_title('discount high, exploitation high')
    rects1 = ax.bar(x - width/2, case1[:,0], width, label='left')
    rects2 = ax.bar(x + width/2, case1[:,1], width, label='right')
    fig, ax = plt.subplots()
    ax.set_title('discount low, exploitation low')
    rects1 = ax.bar(x - width/2, case2[:,0], width, label='left')
    rects2 = ax.bar(x + width/2, case2[:,1], width, label='right')
    plt.show()
