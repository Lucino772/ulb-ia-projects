from mdp import MDP, S, A
from typing import Generic


class ValueIteration(Generic[S, A]):
    def __init__(self, mdp: MDP[S, A], gamma: float):
        # senf.values est nécessaire pour fonctionner avec utils.show_values
        self.values = dict[S, float]()

    def value(self, state: S) -> float:
        """Returns the value of the given state."""

    def policy(self, state: S) -> A:
        """Returns the action that maximizes the Q-value of the given state."""

    def qvalue(self, state: S, action: A) -> float:
        """Returns the Q-value of the given state-action pair based on the state values."""

    def _compute_value_from_qvalues(self, state: S) -> float:
        """
        Returns the value of the given state based on the Q-values.

        This is a private method, meant to be used by the value_iteration method.
        """

    def value_iteration(self, n: int):
        ...
