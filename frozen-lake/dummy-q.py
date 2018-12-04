import gym
from gym.envs.registration import register
import numpy as np
import random as pr
import matplotlib
import matplotlib.pyplot as plt
from util import *

# Register our own environment
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
env.render()

def rargmax(vector):
    m = np.amax(vector)
    # False is interpreted as 0, so this statement means
    # "Find indices whose value is equal to `m`"
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

# Define Q
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 500
eval_bucket = 100

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0  # Total sum of rewards
    done = False

    while not done:
        # Select an action
        action = rargmax(Q[state, :])

        # Perform action - get new state and reward
        new_state, reward, done, info = env.step(action)

        # Update Q function
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state
    
    rList.append(rAll)
    
    # Calculate local success rate.
    if i > 0 and i % eval_bucket == 0:
        local_success_rate = sum(rList[i-eval_bucket:i]) / eval_bucket
        print("Success rate for {} steps: {}".format(eval_bucket, local_success_rate))

print("Success Rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()