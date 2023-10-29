from lle import World, Action
from world_mdp import WorldMDP
from lle import REWARD_AGENT_DIED


def test_state_turn():
    world = WorldMDP(
        World(
            """
    S0 . G
    S1 X X
"""
        )
    )
    s = world.reset()
    assert s.current_agent == 0
    assert s.value == 0
    assert not world.is_final(s)

    actions = [Action.EAST, Action.EAST, Action.EAST, Action.STAY, Action.SOUTH]
    scores = [0, 0, 1, 1, 3]
    assert len(actions) == len(scores), "The test is not well defined."
    for i, (action, score) in enumerate(zip(actions, scores)):
        assert not world.is_final(s)
        s = world.transition(s, action)
        turn = i + 1
        assert s.current_agent == turn % 2
        assert s.value == score
    assert world.is_final(s)


def test_state_score():
    world = WorldMDP(
        World(
            """
    S0 . G
    S1 X X
"""
        )
    )
    s = world.reset()
    assert s.current_agent == 0
    assert s.value == 0
    assert not world.is_final(s)

    actions = [Action.EAST, Action.EAST, Action.EAST, Action.STAY, Action.SOUTH]
    scores = [0, 0, 1, 1, 3]
    assert len(actions) == len(scores), "The test is not well defined."
    for action, score in zip(actions, scores):
        s = world.transition(s, action)
        assert s.value == score


def test_state_current_agent():
    world = WorldMDP(
        World(
            """
    S0 . G
    S1 X X
"""
        )
    )
    s = world.reset()
    assert s.current_agent == 0

    actions = [Action.EAST, Action.EAST, Action.EAST, Action.STAY, Action.SOUTH]
    scores = [0, 0, 1, 1, 3]
    assert len(actions) == len(scores), "The test is not well defined."
    for i, (action, score) in enumerate(zip(actions, scores)):
        s = world.transition(s, action)
        turn = i + 1
        assert s.current_agent == turn % 2
        assert s.value == score


def test_state_is_final():
    world = WorldMDP(World("S0 X"))
    s = world.reset()
    assert not world.is_final(s)
    s = world.transition(s, Action.EAST)
    assert world.is_final(s)


def test_score_death():
    world = WorldMDP(
        World(
            """
    S0  .  X
    S1 L1N X
"""
        )
    )
    s = world.reset()
    assert s.value == 0
    s = world.transition(s, Action.EAST)
    assert s.value == REWARD_AGENT_DIED


def test_score_overwritten():
    world = WorldMDP(
        World(
            """
    S0 G  .  X
    S1 . L1N X
"""
        )
    )
    s = world.reset()
    assert s.value == 0

    # Agent 0
    s = world.transition(s, Action.EAST)
    assert s.value == 1

    # Agent 1
    s = world.transition(s, Action.STAY)

    # Agent 0
    s = world.transition(s, Action.EAST)
    assert s.value == -1, "When Agent 0 dies, the score should be overwritten by -1."


def test_available_actions():
    world = WorldMDP(
        World(
            """
    .  S0 G
    S1 X  X
"""
        )
    )
    s = world.reset()
    available = world.available_actions(s)
    expected = [Action.EAST, Action.STAY, Action.WEST, Action.SOUTH]
    assert all(a in available for a in expected)

    s = world.transition(s, Action.STAY)
    available = world.available_actions(s)
    expected = [Action.EAST, Action.NORTH, Action.EAST]
    assert all(a in available for a in expected)
