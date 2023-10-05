from lle import World
from problem import CornerSearchProblem
from search import astar


def test_corners_reached():
    world = World.from_file("cartes/corners")
    problem = CornerSearchProblem(world)
    solution = astar(problem)
    world.reset()
    corners = set([(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)])
    for action in solution.actions:
        world.step(action)
        agent_pos = world.agents_positions[0]
        if agent_pos in corners:
            corners.remove(agent_pos)
    assert len(corners) == 0, f"The agent did not reach these corners: {corners}"
