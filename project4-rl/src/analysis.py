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
    return Parameters(reward_live=0.8, gamma=0.1, noise=0)


def prefer_close_exit_avoiding_the_cliff() -> Parameters:
    return Parameters(reward_live=0.8, gamma=0.1, noise=0.1)


def prefer_far_exit_following_the_cliff() -> Parameters:
    return Parameters(reward_live=2, gamma=0.4, noise=0)


def prefer_far_exit_avoiding_the_cliff() -> Parameters:
    return Parameters(reward_live=1, gamma=0.5, noise=0.5)


def never_end_the_game() -> Parameters:
    return Parameters(reward_live=1, gamma=0, noise=0)
