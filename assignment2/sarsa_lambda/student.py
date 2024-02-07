import numpy as np
import random


def epsilon_greedy_action(env, Q, state, epsilon):
    action = env.action_space.sample()  # Explore action space
    
    if random.random() < epsilon:
        return action
    else:
        action = np.argmax(Q[state])
        return action

d = {0:"L",1:"D",2:"R",3:"U"}
def getmat(matrix,sh):
    sh = list(sh)
    holes = [[n//6,n%6] for n in sh]
    h_t = np.zeros((5,6))
    for n in holes:
        h_t[n[0]][n[1]] = -1
    a = np.reshape(np.array([np.argmax(matrix[i]) for i in range(matrix.shape[0])]),(5,6))
    l = a.tolist()
    for ri in range(len(l)):
        for ci in range(len(l[ri])):
            if h_t[ri][ci] == -1:
                l[ri][ci] = "H"
            else:
                l[ri][ci] = d.get(l[ri][ci])
                
        print(l[ri])

            

def sarsa_lambda(env, alpha=0.06, gamma=0.999, lambda_= 0.9, initial_epsilon=1.0, n_episodes=20000 ):

    #Q = np.zeros((env.observation_space.n, env.action_space.n))
    Q = np.ones((env.observation_space.n, env.action_space.n))*0.3
    #Q = np.random.rand(env.observation_space.n, env.action_space.n)
    # init epsilon
    epsilon = initial_epsilon
    crt = 0
    received_first_reward = False

    print("TRAINING STARTED")
    print("...")
    sh = set()
    for ep in range(n_episodes):
        E = np.zeros((env.observation_space.n, env.action_space.n)) #eligibility
        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False
        while not done:
            ############## simulate the action
            next_state, reward, done, info, _ = env.step(action)
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)
            d = reward
            
            if not done:
                d += gamma * Q[next_state,next_action]
            d -= Q[state,action] 
            crt += reward
            E[state,action] += 1
            
            Q += alpha * d * E
            E *= gamma * lambda_

            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)
                

            # update current state and action
            if reward == 0 and done:
                sh.add(next_state)
            state = next_state
            action = next_action
            
        # update current epsilon
        if received_first_reward:
            epsilon = 0.999 * epsilon
            
    print("TRAINING FINISHED")

    print(epsilon)
    print(crt)
    print("Q")
    getmat(Q,sh)
    return Q