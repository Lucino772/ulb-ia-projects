import json
from dataclasses import dataclass
from mdp import MDP

Action = str
State = str


@dataclass
class Transition:
    source: State
    action: Action
    destination: State
    reward: float
    probability: float


class GraphMDP(MDP[str, str]):
    @staticmethod
    def from_json(filename: str) -> "GraphMDP":
        with open(filename, "r") as f:
            data = json.load(f)
            start_states = data["start_state"]
            end_states = data["end_states"]
            transitions = {}
            for source, actions in data["transitions"].items():
                transitions[source] = {}
                for action, destinations in actions.items():
                    transitions[source][action] = []
                    for dest in destinations:
                        transitions[source][action].append(
                            Transition(
                                source,
                                action,
                                dest["to"],
                                dest["reward"],
                                dest["probability"],
                            )
                        )
            return GraphMDP(start_states, end_states, transitions)

    def __init__(
        self,
        start_states: list[State],
        end_states: list[State],
        transitions: dict[State, dict[Action, list[Transition]]],
    ):
        self.start_states = set(start_states)
        self.end_states = set(end_states)
        self._transitions = transitions
        self._all_states = self.start_states.union(self.end_states).union(
            set(transitions.keys())
        )

    def is_final(self, state: State) -> bool:
        return state in self.end_states

    def transitions(self, state: State, action: Action) -> list[tuple[State, float]]:
        return [
            (t.destination, t.probability) for t in self._transitions[state][action]
        ]

    def available_actions(self, state: str):
        return self._transitions[state].keys()

    def reward(self, state: str, action: str, new_state) -> float:
        transitions = self._transitions[state][action]
        for t in transitions:
            if t.destination == new_state:
                return t.reward
        raise ValueError(
            f"Invalid transition from {state} to {new_state} with action {action}"
        )

    def states(self):
        return self._all_states
