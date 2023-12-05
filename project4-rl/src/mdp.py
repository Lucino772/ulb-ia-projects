from typing import TypeVar, Generic
from abc import abstractmethod, ABC


A = TypeVar("A")
S = TypeVar("S")


class MDP(ABC, Generic[S, A]):
    """Adversarial Markov Decision Process"""

    @abstractmethod
    def is_final(self, state: S) -> bool:
        """Returns true if the given state is final (i.e. the game is over)."""

    @abstractmethod
    def available_actions(self, state: S) -> list[A]:
        """Returns the list of available actions for the current agent from the given state."""

    @abstractmethod
    def transitions(self, state: S, action: A) -> list[tuple[S, float]]:
        """Returns the list of next states with the probability of reaching it by performing the given action."""

    @abstractmethod
    def states(self) -> list[S]:
        """Returns the list of all states."""

    @abstractmethod
    def reward(self, state: S, action: A, new_state) -> float:
        """Reward function"""
