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
            value = self.run(mdp, next_state, max_depth-1)

            if value >= best_value:
                best_action = action
                best_value = value

        return best_action

minimax = _MinMaxAlgo().execute


class _AlphaBetaAlgo:
    def __init__(self):
        self._alpha_agent = 0
        self._alpha_betas = {}

    def _get_alpha(self):
        if self._alpha_agent not in self._alpha_betas:
            self._alpha_betas[self._alpha_agent] = float("-inf")
        return self._alpha_betas[self._alpha_agent]

    def _set_alpha(self, value: float):
        self._alpha_betas[self._alpha_agent] = max(self._get_alpha(), value)

    def _get_beta(self, agent: int):
        assert agent != self._alpha_agent, f"{agent} is an alpha agent"

        if agent not in self._alpha_betas:
            self._alpha_betas[agent] = float("+inf")
        return self._alpha_betas[agent]

    def _set_beta(self, agent: int, value: float):
        self._alpha_betas[agent] = min(self._get_beta(agent), value)

    @property
    def _max_beta(self):
        _betas = [
            value for key, value in self._alpha_betas.items()
            if key != self._alpha_agent
        ]
        return min(_betas, default=float("+inf"))

    def _min(self, mdp: MDP[A, S], state: S, max_depth: int) -> float:
        best_value = float("+inf")
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            best_value = min(best_value, self.run(mdp, next_state, max_depth-1))

            self._set_beta(state.current_agent, best_value)
            if self._get_alpha() >= self._get_beta(state.current_agent):
                break

        return best_value

    def _max(self, mdp: MDP[A, S], state: S, max_depth: int) -> float:
        best_value = float("-inf")
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            best_value = max(best_value, self.run(mdp, next_state, max_depth-1))

            self._set_alpha(best_value)
            if self._get_alpha() >= self._max_beta:
                break

        return best_value

    def run(self, mdp: MDP[A, S], state: S, max_depth) -> float:
        if mdp.is_final(state) or max_depth == 0:
            return state.value

        if state.current_agent == 0:
            return self._max(mdp, state, max_depth)
        else:
            return self._min(mdp, state, max_depth)

    def execute(self, mdp: MDP[A, S], state: S, max_depth: int) -> A:
        self._alpha_betas.clear()
        if state.current_agent != 0:
            raise ValueError

        best_action = None
        best_value = float("-inf")
        for action in mdp.available_actions(state):
            next_state = mdp.transition(state, action)
            value = self.run(mdp, next_state, max_depth-1)

            if value >= best_value:
                best_action = action
                best_value = value

        return best_action

alpha_beta = _AlphaBetaAlgo().execute

def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> Action:
    ...
