from mdp import MDP, S, A


class RandomWrapper(MDP[S, A]):
    """Wrapper around an MDP such that the action taken can be random with probability p.
    It only changes the `transitions` method. The other methods are unchanged."""

    def __init__(self, mdp: MDP[S, A], p: float):
        super().__init__()
        self.mdp = mdp
        self.p = p
        """The probability of the agent to perform a random action"""

    def is_final(self, state: S) -> bool:
        return self.mdp.is_final(state)

    def available_actions(self, state: S) -> list[A]:
        return self.mdp.available_actions(state)

    def transitions(self, state: S, action: A) -> list[tuple[S, float]]:
        if self.p == 0:
            return self.mdp.transitions(state, action)
        # Create the dictionary of "normal" destinations
        destination_probs = dict(self.mdp.transitions(state, action))
        # The probabilities must be multiplied by (1 - p) because of the wrapper
        for s, prob in destination_probs.items():
            destination_probs[s] = prob * (1 - self.p)
        # Add the random destinations
        available_actions = list(self.available_actions(state))
        n_actions = len(available_actions)
        for action in available_actions:
            for s, prob in self.mdp.transitions(state, action):
                # The probability of taking this action must be multiplied by (p / n_actions)
                prob = (prob * self.p) / n_actions
                destination_probs[s] = destination_probs.get(s, 0.0) + prob
        return list(destination_probs.items())

    def states(self) -> list[S]:
        return self.mdp.states()

    def reward(self, state: S, action: A, new_state) -> float:
        return self.mdp.reward(state, action, new_state)
