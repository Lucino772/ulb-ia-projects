from lle import World
from problem import SimpleSearchProblem, WorldState


def test_goal_state():
    world = World("S0 . X")
    problem = SimpleSearchProblem(world)
    assert not problem.is_goal_state(problem.initial_state)
    end = WorldState([(0, 2)], [])
    assert problem.is_goal_state(end)


def test_successors_one_agent():
    world = World(
        """
        S0 . X
        . . ."""
    )
    problem = SimpleSearchProblem(world)
    successors = list(problem.get_successors(problem.initial_state))
    assert len(successors) == 3
    world.reset()
    available = world.available_actions()[0]
    agent_pos = ((0, 0), (0, 1), (1, 0))
    for state, action, cost in successors:
        assert action[0] in available
        assert state.agents_positions[0] in agent_pos


def test_successors_one_agent_obstacle():
    world = World(
        """
        S0 . X
        @ . ."""
    )
    problem = SimpleSearchProblem(world)
    successors = list(problem.get_successors(problem.initial_state))
    assert len(successors) == 2
    world.reset()
    available = world.available_actions()[0]
    agent_pos = ((0, 0), (0, 1))
    for state, action, cost in successors:
        assert action[0] in available
        assert state.agents_positions[0] in agent_pos


def test_successors_two_agents():
    world = World(
        """
        S0 . X
        S1 . X
        .  . ."""
    )
    problem = SimpleSearchProblem(world)
    successors = list(problem.get_successors(problem.initial_state))
    assert len(successors) == 6
    world.reset()
    available_1 = world.available_actions()[0]
    available_2 = world.available_actions()[1]
    agent_1_pos = ((0, 0), (0, 1))
    agent_2_pos = ((1, 0), (1, 1), (2, 0))
    for state, action, cost in successors:
        assert action[0] in available_1
        assert action[1] in available_2
        assert state.agents_positions[0] in agent_1_pos
        assert state.agents_positions[1] in agent_2_pos
