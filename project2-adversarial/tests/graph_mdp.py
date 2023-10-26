from io import TextIOWrapper
from mdp import MDP, State


GraphAction = str


class GraphState(State):
    def __init__(self, name: str, value: float, current_agent: int):
        super().__init__(value, current_agent)
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphState):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


def parse_diagram(file: TextIOWrapper) -> str:
    diagram = ""
    line = file.readline()
    while line.strip() != '"""':
        diagram += line
        line = file.readline()
    return diagram


def parse_transitions(file: TextIOWrapper, states: dict[str, GraphState], n_agents: int) -> dict[GraphState, dict[GraphAction, GraphState]]:
    transitions = dict[GraphState, dict[GraphAction, GraphState]]()
    line = file.readline().strip()
    while line != '"""':
        state_name, action, next_state_name = line.split()
        if state_name not in states:
            # Should never happen, but just in case
            raise ValueError(f"State {state_name} not found in states.")
            # states[state] = GraphState(state, False, 0, 0)
        state = states[state_name]
        if next_state_name not in states:
            states[next_state_name] = GraphState(next_state_name, 0, 0)
        next_state = states[next_state_name]
        next_state.current_agent = (state.current_agent + 1) % n_agents
        if state not in transitions:
            transitions[state] = dict[GraphAction, GraphState]()
        transitions[state][action] = next_state
        line = file.readline().strip()
    return transitions


def parse_state_values(file: TextIOWrapper, states: dict[str, GraphState]):
    line = file.readline().strip()
    while line != '"""':
        state, value = line.split()
        states[state].value = float(value)
        line = file.readline().strip()


def parse_end_states(line: str, states: dict[str, GraphState]) -> set[GraphState]:
    end_states = set[GraphState]()
    for name in line.split('"')[1].split():
        new_state = GraphState(name, 0, 0)
        states[name] = new_state
        end_states.add(new_state)
    return end_states


class GraphMDP(MDP[GraphAction, GraphState]):
    def __init__(
        self,
        n_agents: int,
        states: list[GraphState],
        transitions: dict[GraphState, dict[GraphAction, GraphState]],
        start_state: GraphState,
        end_states: set[GraphState],
        diagram: str,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.states = states
        self.transitions = transitions
        self.start_state = start_state
        self.diagram = diagram
        self.end_states = end_states
        self.nodes_expanded = 0

    def transition(self, state: GraphState, action: GraphAction) -> GraphState:
        self.nodes_expanded += 1
        return self.transitions[state][action]

    def reset(self) -> GraphState:
        self.nodes_expanded = 0
        return self.start_state

    def available_actions(self, state: GraphState) -> list[GraphAction]:
        return self.transitions[state].keys()

    def is_final(self, state: GraphState) -> bool:
        return state in self.end_states

    @property
    def default_action(self) -> GraphAction:
        return ""

    @staticmethod
    def parse(filename: str) -> "GraphMDP":
        states = dict[str, GraphState]()
        end_states = set[GraphState]()
        with open(filename, "r") as f:
            while line := f.readline():
                line = line.strip()
                if line.startswith("diagram"):
                    diagram = parse_diagram(f)
                elif line.startswith("num_agents"):
                    num_agents = int(line.split('"')[1])
                elif line.startswith("children"):
                    transitions = parse_transitions(f, states, num_agents)
                elif line.startswith("start_state"):
                    start_state = line.split('"')[1].strip()
                    start_state = GraphState(start_state, 0, 0)
                    states[start_state.name] = start_state
                elif line.startswith("win_states") or line.startswith("lose_states"):
                    end_states = end_states.union(parse_end_states(line, states))
                elif line.startswith("evaluation"):
                    parse_state_values(f, states)

        return GraphMDP(
            states=states,
            transitions=transitions,
            start_state=start_state,
            end_states=end_states,
            diagram=diagram,
            n_agents=num_agents,
        )


if __name__ == "__main__":
    mdp = GraphMDP.parse("2-1b-vary-depth.test")
    print(mdp.states)
    print(mdp.transitions)
    print(mdp.start_state)
    print(mdp.diagram)
