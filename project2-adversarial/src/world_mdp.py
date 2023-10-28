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

    def _collected_gems(self, state: WorldMDPState):
        current_world_state = self.world.get_state()
        return (
            current_world_state.gems_collected.count(True)
            - state.world_state.gems_collected.count(True)
        )

    def _agent_just_exit(self, state: WorldMDPState):
        if state.current_agent_pos in self.world.exit_pos:
            return False

        current_agent_pos = self.world.get_state().agents_positions[state.current_agent]
        return current_agent_pos in self.world.exit_pos

    def _get_next_state(self, state: WorldMDPState, step_reward: float):
        next_agent = (state.current_agent+1) % self.world.n_agents
        current_world_state = self.world.get_state()

        if state.current_agent != 0:
            return WorldMDPState(state.value, next_agent, current_world_state)

        if step_reward == lle.REWARD_AGENT_DIED:
            return WorldMDPState(lle.REWARD_AGENT_DIED, next_agent, current_world_state)

        collected_gems_reward = self._collected_gems(state) * lle.REWARD_GEM_COLLECTED
        just_exit_reward = self._agent_just_exit(state) * lle.REWARD_AGENT_JUST_ARRIVED
        return WorldMDPState(
            collected_gems_reward+just_exit_reward,
            next_agent,
            current_world_state
        )


    def reset(self):
        self.n_expanded_states = 0
        self.world.reset()
        self.state = WorldMDPState(0, 0, self.world.get_state())

    def available_actions(self, state: WorldMDPState) -> list[Action]:
        return self.world.available_actions()[state.current_agent]

    def is_final(self, state: WorldMDPState) -> bool:
        if state.value == lle.REWARD_AGENT_DIED:
            return True

        return not any(state.world_state.gems_collected)

    def transition(self, state: WorldMDPState, action: Action) -> WorldMDPState:
        self.n_expanded_states += 1
        actions = self._actions(state.current_agent, action)
        step_reward = self.world.step(actions)
        return self._get_next_state(state, step_reward)


class BetterValueFunction(WorldMDP):
    def transition(self, state: WorldMDPState, action: Action) -> WorldMDPState:
        # Change the value of the state here.
        ...
