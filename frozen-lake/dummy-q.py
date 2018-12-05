import gym
from gym.envs.registration import register
from util import *

# Register our own environment
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
env.render()

# Define Q
Q = np.zeros([env.observation_space.n, env.action_space.n])

observer = LearningObserver()

for i in range(observer.num_episodes):
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

    observer.record(rAll)

    # How fast the model mastered the game
    observer.check_mastered(i)

    # Calculate local success rate.
    observer.observe_local_success_rate(i)

print("Final Q-Table values")
print("LEFT DOWN RIGHT UP")
print(Q)

observer.show_barchart()
