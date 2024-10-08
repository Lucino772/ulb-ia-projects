from itertools import product
from typing import Iterable
from lle import World, WorldState, Action
from mdp import MDP


class WorldMDP(MDP[WorldState, list[Action]]):
    def __init__(self, world: World):
        self.world = world

    def available_actions(self, state: WorldState):
        self.world.set_state(state)
        available = self.world.available_actions()
        return list(product(*available))

    def transitions(
        self, state: WorldState, action: list[Action]
    ) -> list[tuple[WorldState, float]]:
        self.world.set_state(state)
        self.world.step(action)
        return [(self.world.get_state(), 1.0)]

    def is_final(self, state: WorldState) -> bool:
        self.world.set_state(state)
        return self.world.done

    def reward(
        self,
        state: WorldState,
        action: list[Action],
        new_state: WorldState,
    ) -> float:
        # Step the world and check if the new state is the same as the given one
        # If if is not the same, then test all the other available actions.
        self.world.set_state(state)
        actions = [action] + [[a] for a in self.world.available_actions()[0]]
        for a in actions:
            r = self.world.step(a)
            if self.world.get_state() == new_state:
                return r
            self.world.set_state(state)
        raise ValueError("The new state is not reachable from the given state")

    def states(self) -> Iterable[WorldState]:
        all_positions = set(product(range(self.world.height), range(self.world.width)))
        all_positions = all_positions.difference(set(self.world.wall_pos))
        agents_positions = list(product(all_positions, repeat=self.world.n_agents))
        collection_status = list(product([True, False], repeat=self.world.n_gems))

        for pos, collected in product(agents_positions, collection_status):
            s = WorldState(list(pos), list(collected))
            yield s
