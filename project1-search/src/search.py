from dataclasses import dataclass
from typing import Optional
from lle import Action

from problem import SearchProblem


@dataclass
class Solution:
    actions: list[tuple[Action]]

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    ...


def dfs(problem: SearchProblem) -> Optional[Solution]:
    ...


def bfs(problem: SearchProblem) -> Optional[Solution]:
    ...


def astar(problem: SearchProblem) -> Optional[Solution]:
    ...
