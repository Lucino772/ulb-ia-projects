from dataclasses import dataclass
import lle
from lle import World, Action
from mdp import MDP, State


@dataclass
class WorldMDPState(State):
    ...


class WorldMDP(MDP[Action, WorldMDPState]):
    def __init__(self, world: World):
        self.world = world

    def reset(self):
        self.n_expanded_states = 0
        ...

    def available_actions(self, state: WorldMDPState) -> list[Action]:
        ...

    def is_final(self, state: WorldMDPState) -> bool:
        ...

    def transition(self, state: WorldMDPState, action: Action) -> WorldMDPState:
        ...


class BetterValueFunction(WorldMDP):
    def transition(self, state: WorldMDPState, action: Action) -> WorldMDPState:
        # Change the value of the state here.
        ...
