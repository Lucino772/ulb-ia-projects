from lle import Action
from mdp import MDP, S, A


def _min(mdp: MDP[A, S], state: S, max_depth: int) -> float:
    if mdp.is_final(state) or max_depth == 0:
        return state.value

    best_value = float("+inf")

    next_states = [state]
    while len(next_states) > 0:
        current_state = next_states.pop()
        if current_state.current_agent == 0:
            best_value = min(best_value, _max(mdp, current_state, max_depth-1))
        else:
            next_states += [
                mdp.transition(current_state, action)
                for action in mdp.available_actions(current_state)
            ]

    return best_value

def _max(mdp: MDP[A, S], state: S, max_depth: int) -> float:
    if mdp.is_final(state) or max_depth == 0:
        return state.value

    best_value = float("-inf")
    for action in mdp.available_actions(state):
        next_state = mdp.transition(state, action)
        best_value = max(best_value, _min(mdp, next_state, max_depth-1))

    return best_value

def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    if state.current_agent != 0:
        raise ValueError

    best_action = None
    best_value = float("-inf")
    for action in mdp.available_actions(state):
        next_state = mdp.transition(state, action)
        value = _min(mdp, next_state, max_depth-1)

        if value > best_value:
            best_action = action
            best_value = value

    return best_action


def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...


def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> Action:
    ...
