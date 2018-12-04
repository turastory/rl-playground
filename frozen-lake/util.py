# Some utilities for frozen-lake game.

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


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
