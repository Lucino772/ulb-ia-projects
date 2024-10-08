from .world_mdp import WorldMDP
from lle import Action, World, WorldState


class ModifiedRewardWorld(WorldMDP):
    def __init__(self, reward_live: float):
        super().__init__(World.from_file("tests/graphs/cliff"))
        self.world.reset()
        self.reward_live = reward_live

    def reward(
        self, state: WorldState, action: list[Action], new_state: WorldState
    ) -> float:
        reward = super().reward(state, action, new_state)
        # The agent has died
        if reward < 0:
            return -10.0
        # The agent has collected a gem
        # The agent has reached the exit
        if reward > 0:
            # Close exit
            if new_state.agents_positions[0] == (2, 2):
                return 1.0
            # Far exit
            return 10.0
        # The agent just lives (reward should be 0)
        return self.reward_live
