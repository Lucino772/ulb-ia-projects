from dataclasses import dataclass
from typing import Optional
from lle import Action
from priority_queue import PriorityQueue

from problem import SearchProblem


@dataclass
class Solution:
    actions: list[tuple[Action]]

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    ...


def bfs(problem: SearchProblem) -> Optional[Solution]:
    marked_states = set()
    next_states = []

    next_states.append((problem.initial_state, []))
    marked_states.add(problem.initial_state)

    while (len(next_states) > 0):
        initial_state, initial_actions = next_states.pop(0)
        for state, actions, cost in problem.get_successors(initial_state):
            if state not in marked_states:
                total_actions = initial_actions + [actions]
                if problem.is_goal_state(state):
                    return Solution(total_actions)
                else:
                    next_states.append((state, total_actions))
                    marked_states.add(state)

    return None


def dfs(problem: SearchProblem) -> Optional[Solution]:
    marked_states = set()
    next_states = []

    next_states.append((problem.initial_state, []))
    marked_states.add(problem.initial_state)

    while (len(next_states) > 0):
        initial_state, initial_actions = next_states.pop()
        for state, actions, cost in problem.get_successors(initial_state):
            if state not in marked_states:
                total_actions = initial_actions + [actions]
                if problem.is_goal_state(state):
                    return Solution(total_actions)
                else:
                    next_states.append((state, total_actions))
                    marked_states.add(state)

    return None


def astar(problem: SearchProblem) -> Optional[Solution]:
    marked_states = set()
    next_states = PriorityQueue()

    next_states.push((problem.initial_state, [], float("-inf")), float("-inf"))
    marked_states.add(problem.initial_state)

    while (not next_states.is_empty()):
        initial_state, initial_actions, initial_priority = next_states.pop()
        for state, actions, cost in problem.get_successors(initial_state):
            if state not in marked_states:
                total_actions = initial_actions + [actions]
                if problem.is_goal_state(state):
                    return Solution(total_actions)
                else:
                    state_priority = initial_priority + cost + problem.heuristic(state)
                    next_states.push((state, total_actions, state_priority), state_priority)
                    marked_states.add(state)

    return None
