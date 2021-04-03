import gym
import random
import numpy as np

COUNT_TO = 50
class CountEnv(gym.Env):
    def __init__(self):
            self.action_space = gym.spaces.Discrete(COUNT_TO)
            self.observation_space = gym.spaces.Discrete(COUNT_TO)
            self.state = 0

    def step(self, action):
            error = abs((self.state+1)%COUNT_TO - action) / COUNT_TO
            reward = .5 - error

            self.state += 1 if error == 0 else 0
            done = self.state == COUNT_TO
            if done : self.state = 0
            info = {}
            return self.state, reward, done, info

    def reset(self):
            self.state = 0
            return self.state

    def render(self, mode='human', close=False):
        print(self.state)


env = CountEnv()

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = np.zeros([env.observation_space.n, env.action_space.n])
episodes = 1000
for i_episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        # Pick action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        
        # Execute action
        observation, reward, done, info = env.step(action)

        # Evaluate quality of choice
        old_value = q_table[state, action] 
        next_max = np.max(q_table[observation])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        state = observation

env.close()
#print(q_table)

last = 0
for _ in range(COUNT_TO):
    action = np.argmax(q_table[last])
    ok = action == (last+1)%COUNT_TO
    okm = "OK" if ok else "ARR"
    print(f"{last} --> {action} - {okm}")
    last = action

