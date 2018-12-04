import gym
import sys
import tty
import termios
from gym.envs.registration import register

# Handling inputs

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


inkey = _Getch()

arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}

# Register our own environment
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
env.render()

while True:
    # Make an action from the input.
    key = inkey()
    if key not in arrow_keys.keys():
        print("Game aborted..")
        break
    action = arrow_keys[key]

    state, reword, done, info = env.step(action)
    print("State: {}, Action: {}, Reword: {}, Info: {}".format(state, action, reword, info))

    if done:
        print("Finished with reword", reword)
        break
