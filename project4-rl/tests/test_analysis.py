from analysis import (
    prefer_close_exit_avoiding_the_cliff,
    prefer_close_exit_following_the_cliff,
    prefer_far_exit_avoiding_the_cliff,
    prefer_far_exit_following_the_cliff,
    never_end_the_game,
    Parameters,
)
from mdp import MDP
from value_iteration import ValueIteration
from .random_wrapper import RandomWrapper
from .modified_reward_world import ModifiedRewardWorld


def setup_mdp_and_get_path(param: Parameters):
    mdp = ModifiedRewardWorld(param.reward_live)
    start_state = mdp.world.get_state()
    noisy_mdp = RandomWrapper(mdp, param.noise)
    algo = ValueIteration(noisy_mdp, param.gamma)
    algo.value_iteration(100)
    return apply_policy(start_state, mdp, algo)


def apply_policy(state, mdp: MDP, algo: ValueIteration):
    """Apply the policy until the end of the game or a loop is detected. Returns the path taken."""
    path = []
    while not mdp.is_final(state) and state not in path:
        path.append(state)
        action = algo.policy(state)
        transitions = mdp.transitions(state, action)
        assert len(transitions) == 1
        state, _ = transitions[0]
    path.append(state)
    return path


def test_close_exit_following_the_cliff():
    params = prefer_close_exit_following_the_cliff()
    path = setup_mdp_and_get_path(params)
    expected = [(3, 0), (3, 1), (3, 2), (2, 2)]
    assert len(path) == len(expected)
    for exp, state in zip(expected, path):
        assert state.agents_positions[0] == exp


def test_close_exit_avoiding_the_cliff():
    params = prefer_close_exit_avoiding_the_cliff()
    path = setup_mdp_and_get_path(params)
    expected = [(3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert len(path) == len(expected)
    for exp, state in zip(expected, path):
        assert state.agents_positions[0] == exp


def test_far_exit_following_the_cliff():
    params = prefer_far_exit_following_the_cliff()
    path = setup_mdp_and_get_path(params)
    expected = [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (2, 4)]
    assert len(path) == len(expected)
    for exp, state in zip(expected, path):
        assert state.agents_positions[0] == exp


def test_far_exit_avoiding_the_cliff():
    params = prefer_far_exit_avoiding_the_cliff()
    path = setup_mdp_and_get_path(params)
    assert len(path) == 10
    # This is only the start of the path because the second half could vary
    expected = [(3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]
    for exp, state in zip(expected, path):
        assert state.agents_positions[0] == exp


def test_never_end():
    params = never_end_the_game()
    path = setup_mdp_and_get_path(params)
    # Check that there is a loop in the path
    assert path[-1] in path[:-1]
