from search import Solution
from problem import SimpleSearchProblem


def check_world_done(problem: SimpleSearchProblem, solution: Solution):
    world = problem.world
    world.reset()
    for action in solution.actions:
        world.step(action)
    assert world.done
