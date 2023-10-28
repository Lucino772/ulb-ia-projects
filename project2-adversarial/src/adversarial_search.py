from lle import Action
from mdp import MDP, S, A

class _MinMaxAlgo:
    def _min(self, mdp: MDP[A, S], state: S, max_depth: int) -> float:
        best_value = float("+inf")
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            best_value = min(best_value, self.run(mdp, next_state, max_depth-1))

        return best_value

    def _max(self, mdp: MDP[A, S], state: S, max_depth: int) -> float:
        best_value = float("-inf")
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            best_value = max(best_value, self.run(mdp, next_state, max_depth-1))

        return best_value

    def run(self, mdp: MDP[A, S], state: S, max_depth) -> float:
        if mdp.is_final(state) or max_depth == 0:
            return state.value

        if state.current_agent == 0:
            return self._max(mdp, state, max_depth)
        else:
            return self._min(mdp, state, max_depth)

    def execute(self, mdp: MDP[A, S], state: S, max_depth: int) -> A:
        if state.current_agent != 0:
            raise ValueError

        best_action = None
        best_value = float("-inf")
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            value = self.run(mdp, next_state, max_depth)

            if value >= best_value:
                best_action = action
                best_value = value

        return best_action

minimax = _MinMaxAlgo().execute


def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...


def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> Action:
    ...
