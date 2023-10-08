from abc import ABC, abstractmethod
from typing import List, MutableMapping, Tuple, Iterable, Generic, TypeVar
from lle import World, Action, WorldState, Position
import itertools
import copy
import math

from dataclasses import dataclass

T = TypeVar("T")

def manhattan_distance(start: "Position", end: "Position"):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

def euclidian_distance(start: "Position", end: "Position"):
    return math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)

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

@dataclass
class CornerProblemState:
    world_state: "WorldState"
    agents_corners_passed: List[MutableMapping["Position", bool]]

    @property
    def agents_positions(self):
        return self.world_state.agents_positions

    def _get_agents_corners_tuples(self):
        return tuple([
            tuple(corners.items())
            for corners in self.agents_corners_passed
        ])

    def __eq__(self, other: object):
        if not isinstance(other, CornerProblemState):
            return False

        if self.world_state != other.world_state:
            return False

        if self._get_agents_corners_tuples() != other._get_agents_corners_tuples():
            return False

        return True

    def __hash__(self) -> int:
        return hash((
            self.world_state,
            self._get_agents_corners_tuples()
        ))

class CornerSearchProblem(SearchProblem[CornerProblemState]):
    def __init__(self, world: World):
        super().__init__(world)
        self.corners = [(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)]
        self.initial_state = self._convert_world_state(self.world.get_state())

    def _convert_world_state(self, state: "WorldState", agents_corners_passed: List[MutableMapping["Position", bool]] = None):
        if agents_corners_passed is None:
            agents_corners_passed=[
                { corner: False for corner in self.corners }
                for _ in range(self.world.n_agents)
            ]

        return CornerProblemState(state, copy.deepcopy(agents_corners_passed))

    def is_goal_state(self, state: CornerProblemState) -> bool:
        # We check if every agent passed each corner
        all_corners_passed = all([all(corners_passed.values()) for corners_passed in state.agents_corners_passed])
        if not all_corners_passed:
            return False

        # We check if every agent is currently in one of the exit
        exit_pos = self.world.exit_pos[:]
        for agent_pos in state.agents_positions:
            if agent_pos in exit_pos:
                exit_pos.remove(agent_pos)

        return len(exit_pos) == 0

    def _apply_actions(self, actions: Tuple["Action", ...], prev_state: "CornerProblemState"):
        self.world.step(actions)
        state = self._convert_world_state(self.world.get_state(), prev_state.agents_corners_passed)
        self.world.set_state(prev_state.world_state)

        # Check agent passed corner
        for agent_id, agent_pos in enumerate(state.agents_positions):
            if agent_pos in self.corners and not state.agents_corners_passed[agent_id][agent_pos]:
                state.agents_corners_passed[agent_id][agent_pos] = True

        return state, actions, 0

    def get_successors(self, state: CornerProblemState) -> Iterable[Tuple[CornerProblemState, Action, float]]:
        initial_state = self.world.get_state()
        self.nodes_expanded += 1

        states = []
        self.world.set_state(state.world_state)
        if not self.world.done:
            states = [
                self._apply_actions(actions, state)
                for actions in itertools.product(*self.world.available_actions())
            ]

        self.world.set_state(initial_state)
        return states

    def heuristic(self, state: CornerProblemState) -> float:
        available_exit_pos = [
            pos for pos in self.world.exit_pos
            if pos not in state.agents_positions
        ]

        distance_sum = 0
        for agent_id, agent_pos in enumerate(state.agents_positions):
            corners_to_reach = [
                (corner_pos, corner_state)
                for corner_pos, corner_state in state.agents_corners_passed[agent_id].items()
                if corner_state is False
            ]
            if len(corners_to_reach) > 0:
                # Calculate the distance between agent and corners that have not been
                # reached yet and take the smallest distance.
                distance_sum += min([
                    manhattan_distance(agent_pos, corner_pos)
                    for corner_pos, _ in corners_to_reach
                ])
            else:
                # If all corners have been reached, compute distance with exit
                if agent_pos in self.world.exit_pos:
                    distance_sum += 0
                else:
                    distance_sum += min([
                        manhattan_distance(agent_pos, exit_pos)
                        for exit_pos in available_exit_pos
                    ])

        return distance_sum

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
