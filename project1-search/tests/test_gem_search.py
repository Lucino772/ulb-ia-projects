from lle import World
from problem import GemSearchProblem
from search import astar

from .utils import check_world_done


def test_gems_collected():
    world = World.from_file("cartes/gems")
    problem = GemSearchProblem(world)
    solution = astar(problem)
    check_world_done(problem, solution)
    if world.n_gems != world.gems_collected:
        raise AssertionError("Your is_goal_state method is likely erroneous beacuse some gems have not been collected")
