from dataclasses import dataclass


@dataclass
class Parameters:
    reward_live: float
    """Reward for living at each time step"""
    gamma: float
    """Discount factor"""
    noise: float
    """Probability of taking a random action instead of the chosen one"""


def prefer_close_exit_following_the_cliff() -> Parameters:
    return Parameters(...)


def prefer_close_exit_avoiding_the_cliff() -> Parameters:
    return Parameters(...)


def prefer_far_exit_following_the_cliff() -> Parameters:
    return Parameters(...)


def prefer_far_exit_avoiding_the_cliff() -> Parameters:
    return Parameters(...)


def never_end_the_game() -> Parameters:
    return Parameters(...)
