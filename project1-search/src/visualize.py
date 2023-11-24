from lle import World

import cv2
import search
from problem import CornerSearchProblem

DISPLAY = True
ALGORITHMS = {
    # "dfs": search.dfs,
    # "bfs": search.bfs,
    "astar": search.astar
}

def test(world: "World"):
    results = dict.fromkeys(ALGORITHMS.keys(), None)
    for algo, method in ALGORITHMS.items():
        problem = CornerSearchProblem(world)
        results[algo] = method(problem), problem.nodes_expanded

    world.reset()
    return results

def display_world(name: str, world: "World"):
    if DISPLAY:
        cv2.imshow(name, world.get_image())
        cv2.waitKey(0)
        cv2.waitKey(1)

def display_solution(name: str, world: "World", solution):
    world.reset()
    display_world(name, world)

    for action in solution.actions:
        world.step(action)
        display_world(name, world)

world = World.from_file("cartes/corners")
world.reset()

display_world("initial", world)

results = test(world)
for algo, (solution, expansion) in results.items():
    print(algo, solution.n_steps, expansion)
    display_solution(algo, world, solution)
