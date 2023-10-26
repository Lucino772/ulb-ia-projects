from lle import Action
from mdp import MDP, S, A


def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...


def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...


def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> Action:
    ...
