from lle import World, Action
from adversarial_search import minimax
from world_mdp import WorldMDP
from .graph_mdp import GraphMDP


def test_raise_value_error():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    s0 = mdp.reset()
    s = mdp.transition(s0, "Right")
    try:
        minimax(mdp, s, 2)
        assert False, "Should raise ValueError"
    except ValueError:
        assert True


def test_graph_mdp():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    assert minimax(mdp, mdp.reset(), 1) == "Right"
    assert minimax(mdp, mdp.reset(), 2) == "Left"
    assert minimax(mdp, mdp.reset(), 3) == "Right"


def test_minimax_two_agents():
    world = WorldMDP(
        World(
            """
        .  L1W
        S0 S1
        X  X"""
        )
    )
    world.reset()
    action = minimax(world, world.reset(), 5)
    assert action in [Action.SOUTH, Action.STAY]

    world.reset()
    action = minimax(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_minimax_greedy():
    world = WorldMDP(
        World(
            """
S0 . G G
G  @ @ @
.  . X X
S1 . . .
"""
        )
    )
    action = minimax(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_minimax_greedy_3steps():
    world = WorldMDP(
        World(
            """
S0 . G G
G  @ @ @
.  . X X
S1 . . .
"""
        )
    )
    action = minimax(world, world.reset(), 5)
    assert (
        action == Action.EAST
    ), "When the agent sees 5 steps in the future, it should realise that going east is the best option to get both gems."


def test_minimax_greedy_5steps():
    world = WorldMDP(
        World(
            """
S0 . G G
G  @ @ @
.  . X X
S1 . . .
"""
        )
    )
    action = minimax(world, world.reset(), 10)
    assert (
        action == Action.SOUTH
    ), "When the agent sees 10 steps in the future, it should realise that it should first take the bottom gem before the other two."


def test_three_agents():
    world = WorldMDP(
        World(
            """
        S0 . G G .
        G  @ @ @ .
        .  . X X X
        S1 . . . S2
"""
        )
    )
    action = minimax(world, world.reset(), 1)
    assert action == Action.SOUTH


def test_three_agents2():
    """In this test, Agent 2 should take the gem on top of him
    in order to prevent Agent 0 from getting it, even if Agent 2
    could deny two gems by going left."""
    world = WorldMDP(
        World(
            """
        .  . . . G G S0
        .  . . @ @ @ G
        S2 . . X X X G
        .  . . . G G S1
"""
        )
    )
    action = minimax(world, world.reset(), 1)
    assert action == Action.SOUTH

    action = minimax(world, world.reset(), 3)
    assert action == Action.WEST

    action = minimax(world, world.reset(), 7)
    assert action == Action.SOUTH


def test_two_agents_laser():
    world = WorldMDP(
        World(
            """
        S0 G  .  X
        .  .  .  .
        X L1N S1 .
"""
        )
    )
    action = minimax(world, world.reset(), 3)
    assert action == Action.SOUTH

    action = minimax(world, world.reset(), 2)
    assert action != Action.EAST
