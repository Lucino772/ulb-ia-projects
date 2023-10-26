from dataclasses import dataclass
import lle
from lle import World, Action
from mdp import MDP, State


@dataclass
class MyWorldState(State):
    ...


class WorldMDP(MDP[Action, MyWorldState]):
    def __init__(self, world: World):
        self.world = world

    def reset(self):
        self.n_expanded_states = 0
        ...

    def available_actions(self, state: MyWorldState) -> list[Action]:
        ...

    def is_final(self, state: MyWorldState) -> bool:
        ...

    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        ...


class BetterValueFunction(WorldMDP):
    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        # Change the value of the state here.
        ...
