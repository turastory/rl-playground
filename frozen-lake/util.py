# Some utilities for frozen-lake game.
import numpy as np
import random as pr
import matplotlib.pyplot as plt

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def rargmax(vector):
    m = np.amax(vector)
    # False is interpreted as 0, so this statement means
    # "Find indices whose value is equal to `m`"
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


def interpret_action(action):
    if action == LEFT:
        return "LEFT"
    elif action == RIGHT:
        return "RIGHT"
    elif action == DOWN:
        return "DOWN"
    elif action == UP:
        return "UP"
    else:
        return "UNKNOWN"


def show_barchart(showingList):
    plt.bar(range(len(showingList)), showingList, color="blue")
    plt.show()


class LearningObserver:
    def __init__(self):
        self.reset()

    def reset(self, ):
        self.reward_list = []
        self.mastered = False
        self.master_continuous_success = 10
        self.master_threshold = 0.95
        self.num_episodes = 500
        self.num_local_eval_episodes = 50

    def record(self, reward):
        self.reward_list.append(reward)

    def observe_local_success_rate(self, step):
        if step > 0 and step % self.num_local_eval_episodes == 0:
            local_success_rate = sum(
                self.reward_list[step - self.num_local_eval_episodes:step]) / self.num_local_eval_episodes
            print("Success rate for {} steps: {}".format(
                self.num_local_eval_episodes, local_success_rate))

    def observe_total_success_rate(self):
        print("Success Rate: " + str(sum(self.reward_list) / self.num_episodes))

    def check_mastered(self, step):
        def isMastered(item):
            return item > self.master_threshold

        if not self.mastered and step >= self.master_continuous_success:
            start = step - self.master_continuous_success
            if all(map(isMastered, self.reward_list[start:step])):
                self.mastered = True
                print("Mastered the game at step {}".format(step))

    def show_barchart(self):
        show_barchart(self.reward_list)
