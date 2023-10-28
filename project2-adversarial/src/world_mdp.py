from dataclasses import dataclass
import lle
from lle import World, Action, WorldState
from mdp import MDP, State


@dataclass
class WorldMDPState(State):
    world_state: WorldState

    @property
    def current_agent_pos(self):
        return self.world_state.agents_positions[self.current_agent]


class WorldMDP(MDP[Action, WorldMDPState]):
    def __init__(self, world: World):
        self.world = world
        self.state = WorldMDPState(0, 0, self.world.get_state())

    def _actions(self, agent: int, action: Action):
        return [
            action if agent == _agent else Action.STAY
            for _agent in range(self.world.n_agents)
        ]

    def _get_next_state(self, state: WorldMDPState, step_reward: float):
        next_agent = (state.current_agent + 1) % self.world.n_agents
        current_world_state = self.world.get_state()

        if state.current_agent != 0:
            return WorldMDPState(state.value, next_agent, current_world_state)

        if step_reward == lle.REWARD_AGENT_DIED:
            return WorldMDPState(lle.REWARD_AGENT_DIED, next_agent, current_world_state)

        return WorldMDPState(
            state.value + step_reward,
            next_agent,
            current_world_state,
        )

    def reset(self):
        self.n_expanded_states = 0
        self.world.reset()
        self.state = WorldMDPState(0, 0, self.world.get_state())
        return self.state

    def available_actions(self, state: WorldMDPState) -> list[Action]:
        current_state = self.world.get_state()
        self.world.set_state(state.world_state)

        # INFO: If world is done, we make sure no actions are returned
        actions = []
        if not self.world.done:
            actions = self.world.available_actions()[state.current_agent]

        self.world.set_state(current_state)
        return actions

    def is_final(self, state: WorldMDPState) -> bool:
        if state.value == lle.REWARD_AGENT_DIED:
            return True

        if not all(state.world_state.gems_collected):
            return False

        # We check if every agent is currently in one of the exit
        exit_pos = self.world.exit_pos[:]
        for agent_pos in state.world_state.agents_positions:
            if agent_pos in exit_pos:
                exit_pos.remove(agent_pos)

        return len(exit_pos) == 0

    def transition(self, state: WorldMDPState, action: Action) -> WorldMDPState:
        self.n_expanded_states += 1
        self.world.set_state(state.world_state)
        step_reward = self.world.step(self._actions(state.current_agent, action))
        return self._get_next_state(state, step_reward)


class BetterValueFunction(WorldMDP):
    def transition(self, state: WorldMDPState, action: Action) -> WorldMDPState:
        # Change the value of the state here.
        ...
