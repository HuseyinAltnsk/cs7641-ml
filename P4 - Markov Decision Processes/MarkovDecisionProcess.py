import numpy as np

class MarkovDecisionProcess():
    def __init__(self, num_states, num_actions, dynamics_fn):
        """
        Initializes an object representing a Markov Decision Process.
        Assumes the reward is deterministic for a given state.
        """
        assert num_states > 0
        assert num_actions > 0
        self.num_states = num_states
        self.num_actions = num_actions
        # P[s][a] represents a list of possible transistions given state s and a.
        # each transistion is expected to be as: [prob_next_state, next_state, reward, is_terminal]
        self.P = dynamics_fn
        # sanity checks
        self.__verify()
    
    def __verify(self):
        assert len(self.P) == self.num_states
        for s in self.P.keys():
             assert len(self.P[s]) == self.num_actions
        for s in self.P.keys():
            for a in self.P[s].keys():
                transitions = self.P[s][a]
                p_sum = sum([t[0] for t in transitions])
                assert p_sum <= 1 and p_sum > 0.99
    
if __name__ == '__main__':
    ns, na = 4, 2
    P = {s : {a : [((1/na), (s+1)%4, None, None) for a in range(na)] for a in range(na)} for s in range(ns)}
    mdp = MarkovDecisionProcess(4, 2, P)
