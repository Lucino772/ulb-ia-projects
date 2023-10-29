from typing import Callable, Tuple
from lle import Action
from mdp import MDP, S, A
from functools import wraps

def _ensure_agent0(algo_func: Callable[[MDP[A, S], S, int], Tuple[float, A]]):
    @wraps(algo_func)
    def _wrapper(mdp: MDP[A, S], state: S, depth: int) -> A:
        if state.current_agent != 0:
            raise ValueError
        return algo_func(mdp, state, depth)[1]

    return _wrapper


def _iter_states(mdp: MDP[A, S], state: S):
    for action in mdp.available_actions(state):
        _state = mdp.transition(state, action)
        yield _state, action


def _minimax(mdp: MDP[A, S], state: S, depth: int) -> tuple[float, A]:
    if (depth == 0) or (mdp.is_final(state)):
        return state.value, None

    if state.current_agent == 0:
        best_value = float("-inf")
        best_action = None

        for next_state, action in _iter_states(mdp, state):
            value, _ = _minimax(mdp, next_state, depth - 1)
            if value > best_value:
                best_value = value
                best_action = action
    else:
        best_value = float("+inf")
        best_action = None

        for next_state, action in _iter_states(mdp, state):
            next_depth = depth - (not min(next_state.current_agent, 1))

            value, _ = _minimax(mdp, next_state, next_depth)
            if value < best_value:
                best_value = value
                best_action = action

    return best_value, best_action


def _alpha_beta(
    mdp: MDP[A, S],
    state: S,
    depth: int,
    alpha: float = float("-inf"),
    beta: float = float("+inf"),
) -> tuple[float, A]:
    if (depth == 0) or (mdp.is_final(state)):
        return state.value, None

    if state.current_agent == 0:
        best_value = float("-inf")
        best_action = None

        for next_state, action in _iter_states(mdp, state):
            value, _ = _alpha_beta(mdp, next_state, depth - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action

            # INFO: alpha-beta pruning
            alpha = max(alpha, value)
            if value >= beta:
                break
    else:
        best_value = float("+inf")
        best_action = None

        for next_state, action in _iter_states(mdp, state):
            next_depth = depth - (not min(next_state.current_agent, 1))

            value, _ = _alpha_beta(mdp, next_state, next_depth, alpha, beta)
            if value < best_value:
                best_value = value
                best_action = action

            # INFO: alpha-beta pruning
            beta = min(beta, value)
            if value <= alpha:
                break

    return best_value, best_action


def _expectimax(mdp: MDP[A, S], state: S, depth: int) -> Action:
    if (depth == 0) or (mdp.is_final(state)):
        return state.value, None

    if state.current_agent == 0:
        best_value = float("-inf")
        best_action = None

        for next_state, action in _iter_states(mdp, state):
            value, _ = _expectimax(mdp, next_state, depth - 1)
            if value > best_value:
                best_value = value
                best_action = action
    else:
        best_value = 0
        best_action = None

        states = list(_iter_states(mdp, state))
        if len(states) != 0:
            probability = 1 / len(states)

            for next_state, action in states:
                next_depth = depth - (not min(next_state.current_agent, 1))
                best_value += probability * _expectimax(mdp, next_state, next_depth)[0]

    return best_value, best_action


minimax = _ensure_agent0(_minimax)
alpha_beta = _ensure_agent0(_alpha_beta)
expectimax = _ensure_agent0(_expectimax)
