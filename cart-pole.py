import gym
import numpy as np

env = gym.make("CartPole-v1")

observation = env.reset()

print (env.action_space)
print (env.observation_space)

max_position = 4.8
max_angle = 0.418
thrust = .5
thrust_period = 5
same = 0

right = False
for i in range(1000):
    env.render()

    action = 1 if right else 0

    # Execute action
    observation, reward, done, info = env.step(action)

    # Find worst variable and move to fix that
    cart_pos = observation[0]
    pole_angle = observation[2]

    pos_limit = abs(cart_pos) / max_position
    angle_limit = abs(pole_angle) / max_angle

    last = right
    if pos_limit > angle_limit: # Position is bad, fix that
        right = cart_pos < 0
    else: # Pole angle is bad
        right = pole_angle > 0

    if last == right:
        same += 1
        if same / thrust_period > thrust:
            right = not right
            same = 0
        if i % thrust_period == 0:
            same = 0

    if done:
        observation = env.reset()

env.close()