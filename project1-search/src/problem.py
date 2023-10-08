from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Generic, TypeVar
from lle import World, Action, WorldState, Position
import itertools


T = TypeVar("T")

def manhattan_distance(start: "Position", end: "Position"):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

class SearchProblem(ABC, Generic[T]):
    """
    A Search Problem is a problem that can be solved by a search algorithm.

    The generic parameter T is the type of the problem state, which must inherit from WorldState.
    """

    def __init__(self, world: World):
        self.world = world
        world.reset()
        self.initial_state = world.get_state()
        self.nodes_expanded = 0

    @abstractmethod
    def is_goal_state(self, problem_state: T) -> bool:
        """Whether the given state is the goal state"""

    @abstractmethod
    def get_successors(self, state: T) -> Iterable[Tuple[T, Tuple[Action, ...], float]]:
        """
        Yield all possible states that can be reached from the given world state.
        Returns
            - the new problem state
            - the joint action that was taken to reach it
            - the cost of taking the action
        """

    def heuristic(self, problem_state: T) -> float:
        return 0.0


class SimpleSearchProblem(SearchProblem[WorldState]):
    def is_goal_state(self, state: WorldState) -> bool:
        # We check if every agent is currently in one of the exit
        exit_pos = self.world.exit_pos[:]
        for agent_pos in state.agents_positions:
            if agent_pos in exit_pos:
                exit_pos.remove(agent_pos)

        return len(exit_pos) == 0

    def _apply_actions(self, actions: Tuple["Action", ...], reset_state: "WorldState"):
        self.world.step(actions)
        state = self.world.get_state()
        self.world.set_state(reset_state)
        return state

    def get_successors(self, state: WorldState) -> Iterable[Tuple["WorldState", Tuple["Action", ...], float]]:
        initial_state = self.world.get_state()
        self.nodes_expanded += 1

        states = []
        self.world.set_state(state)
        if not self.world.done:
            states = [
                (self._apply_actions(actions, state), actions, 0)
                for actions in itertools.product(*self.world.available_actions())
            ]

        self.world.set_state(initial_state)
        return states

    def heuristic(self, state: WorldState) -> float:
        """Manhattan distance for each agent to the closest exit"""
        available_exit_pos = [
            pos for pos in self.world.exit_pos
            if pos not in state.agents_positions
        ]

        distance_sum = 0
        for agent_pos in state.agents_positions:
            if agent_pos in self.world.exit_pos:
                distance_sum += 0
            else:
                distance_sum += min([
                    manhattan_distance(agent_pos, exit_pos)
                    for exit_pos in available_exit_pos
                ])

        return distance_sum


class CornerProblemState:
    ...


class CornerSearchProblem(SearchProblem[CornerProblemState]):
    def __init__(self, world: World):
        super().__init__(world)
        self.corners = [(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)]
        self.initial_state = ...

    def is_goal_state(self, state: CornerProblemState) -> bool:
        raise NotImplementedError()

    def heuristic(self, problem_state: CornerProblemState) -> float:
        raise NotImplementedError()

    def get_successors(self, state: CornerProblemState) -> Iterable[Tuple[CornerProblemState, Action, float]]:
        self.nodes_expanded += 1
        raise NotImplementedError()


class GemProblemState:
    ...


class GemSearchProblem(SearchProblem[GemProblemState]):
    def __init__(self, world: World):
        super().__init__(world)
        self.initial_state = ...

    def is_goal_state(self, state: GemProblemState) -> bool:
        raise NotImplementedError()

    def heuristic(self, state: GemProblemState) -> float:
        """The number of uncollected gems"""
        raise NotImplementedError()

    def get_successors(self, state: GemProblemState) -> Iterable[Tuple[GemProblemState, Action, float]]:
        self.nodes_expanded += 1
        raise NotImplementedError()
