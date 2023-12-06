from mdp import MDP, S, A
from typing import Generic


class ValueIteration(Generic[S, A]):
    def __init__(self, mdp: MDP[S, A], gamma: float):
        # self.values est nécessaire pour fonctionner avec utils.show_values
        self.values = dict[S, float]()

        self._mdp = mdp
        self._gamma = gamma

    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        if state not in self.values:
            self.values[state] = 0

        return self.values[state]

    def policy(self, state: S) -> A:
        """Returns the action that maximizes the Q-value of the given state."""
        return max(
            [
                (self.qvalue(state, action), action)
                for action in self._mdp.available_actions(state)
            ],
            key=lambda item: item[0],
        )[1]

    def qvalue(self, state: S, action: A) -> float:
        """Returns the Q-value of the given state-action pair based on the state values."""
        return sum(
            [
                probability
                * (
                    self._mdp.reward(state, action, next_state)
                    + (self._gamma * self.value(next_state))
                )
                for next_state, probability in self._mdp.transitions(state, action)
            ]
        )

    def _compute_value_from_qvalues(self, state: S) -> float:
        """
        Returns the value of the given state based on the Q-values.

        This is a private method, meant to be used by the value_iteration method.
        """
        if self._mdp.is_final(state):
            return 0

        return max(
            [
                self.qvalue(state, action)
                for action in self._mdp.available_actions(state)
            ]
        )

    def value_iteration(self, n: int):
        for _ in range(n):
            for state in self._mdp.states():
                self.values[state] = self._compute_value_from_qvalues(state)
