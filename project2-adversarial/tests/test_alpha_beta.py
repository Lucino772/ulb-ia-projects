from lle import World, Action
from adversarial_search import alpha_beta
from world_mdp import WorldMDP
from .graph_mdp import GraphMDP


def test_raise_value_error():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    s0 = mdp.reset()
    s = mdp.transition(s0, "Right")
    try:
        alpha_beta(mdp, s, 2)
        assert False, "Should raise ValueError"
    except ValueError:
        assert True


def test_alpha_beta_graph_mdp():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    assert alpha_beta(mdp, mdp.reset(), 1) == "Right"
    assert mdp.nodes_expanded == 2

    assert alpha_beta(mdp, mdp.reset(), 2) == "Left"
    assert mdp.nodes_expanded == 5

    assert alpha_beta(mdp, mdp.reset(), 3) == "Right"
    assert mdp.nodes_expanded == 9


def test_alpha_beta_two_agents():
    world = WorldMDP(
        World(
            """
        .  L1W
        S0 S1
        X  X"""
        )
    )
    action = alpha_beta(world, world.reset(), 5)
    assert action in [Action.SOUTH, Action.STAY]
    assert world.n_expanded_states <= 30

    action = alpha_beta(world, world.reset(), 1)
    assert action == Action.SOUTH
    assert world.n_expanded_states <= 4


def test_alpha_beta_greedy():
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
    action = alpha_beta(world, world.reset(), 1)
    assert action == Action.SOUTH
    assert world.n_expanded_states <= 4


def test_alpha_beta_greedy_3steps():
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
    action = alpha_beta(world, world.reset(), 5)
    assert (
        action == Action.EAST
    ), "When the agent sees 5 steps in the future, it should realise that going east is the best option to get both gems."
    assert world.n_expanded_states <= 148


def test_alpha_beta_greedy_5steps():
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
    action = alpha_beta(world, world.reset(), 10)
    assert (
        action == Action.SOUTH
    ), "When the agent sees 10 steps in the future, it should realise that it should first take the bottom gem before the other two."
    assert world.n_expanded_states <= 5869


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
    action = alpha_beta(world, world.reset(), 1)
    assert action == Action.SOUTH
    assert world.n_expanded_states <= 4


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
    action = alpha_beta(world, world.reset(), 1)
    assert action == Action.SOUTH
    assert world.n_expanded_states <= 4

    world.reset()
    action = alpha_beta(world, world.reset(), 3)
    assert action == Action.WEST
    assert world.n_expanded_states <= 116

    world.reset()
    action = alpha_beta(world, world.reset(), 7)
    assert action == Action.SOUTH
    assert world.n_expanded_states <= 31724


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
    action = alpha_beta(world, world.reset(), 3)
    assert action == Action.SOUTH
    assert world.n_expanded_states <= 27

    world.reset()
    action = alpha_beta(world, world.reset(), 2)
    assert action != Action.EAST
    assert world.n_expanded_states <= 8
