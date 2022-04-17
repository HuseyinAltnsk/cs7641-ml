# Adjusted code taken from: https://medium.com/swlh/frozen-lake-as-a-markov-decision-process-1692815ecfd1

import random
import numpy as np

class Agent():
    def __init__(self, mdp, discount_rate=1.0, theta=0.05):
        """
        Agent to solve worlds that are modelled using finite MDPs.
        Note: rewards are assumed to be deterministic for any given state.
        """
        self.mdp = mdp
        self.value_fn = [0] * self.mdp.num_states
        self.theta = theta
        num_a = self.mdp.num_actions
        num_s = self.mdp.num_states
        # random deterministic policy
        self.policy = []
        for s in range(num_s):
            p_s = [0] * num_a
            random_action = random.randint(0, num_a-1)
            p_s[random_action] = 1
            self.policy.append(p_s)
        self.discount_rate = discount_rate

    def get_action(self, s):
        """
        Using self.policy, return an action for state s.
        Supports stochastic policies.
        """
        current_action_prob_dist = self.policy[s]
        action = np.random.choice(list(range(self.mdp.num_actions)), 1, p=current_action_prob_dist)
        return action[0]

    def evaluate_policy(self):
        """
        Policy evaluation for all states.
        Uses self.theta to determine stopping distance.
        This is not a pure function as it updates the value function for all states.
        """
        delta = self.theta + 1
        while delta > self.theta:
            delta = 0
            for s in range(self.mdp.num_states):
                value_s_old = self.value_fn[s]
                self.value_fn[s] = self.evaluate_policy_for_state(s)
                delta = max(delta, np.abs(value_s_old - self.value_fn[s]))
        
    
    def evaluate_policy_for_state(self, s, action=None):
        """
        Policy evaluation for state s; action is optional.
        If action is None, it estimates state value function, v(s).
        If action is given, it estimates state-action value function q(s,a).
        This is a pure function with respect to class members.
        """
        assert 0 <= s < self.mdp.num_states
        assert (action is None) or (0 <= action < self.mdp.num_actions)
        value_s = 0
        if action is None:
            action = self.get_action(s)
        for transition in self.mdp.P[s][action]:
            # transition informs the next state, probability of it, reward and if state is terminal
            pr_s_ = transition[0] # prob of next state s_ given s and action
            s_ = transition[1] # value of next state s_ given s and action
            r_s_ = transition[2] # reward of next state s_ given s and action
            value_s += pr_s_ * (r_s_ + self.discount_rate * self.value_fn[s_])
        return value_s

    def improve_policy(self, debug=False):
        """
        Using the current value function, improve the existing policy.
        Returns a boolean indicating whether the policy has changed.
        This is not a pure function, as it modifies the member self.policy
        """
        is_policy_stable = True
        debug and print(self.policy)
        for s in range(self.mdp.num_states):
            current_action = self.get_action(s)
            action_max = self.improve_policy_for_state(s)
            debug and print(f"State {s}, Action {current_action}, New action {action_max}")
            if action_max != current_action:
                is_policy_stable = False
                self.policy[s][current_action] = 0
                self.policy[s][action_max] = 1
        return is_policy_stable
        

    def improve_policy_for_state(self, s):
        """
        Using the current value function, returns an action that improves the existing policy at state s.
        Returns a boolean indicating whether the policy has changed.
        This is a pure function with respect to class members.
        """
        assert 0 <= s < self.mdp.num_states
        
        v_s_max = float("-inf")
        for action in range(self.mdp.num_actions):
            v_s = self.evaluate_policy_for_state(s, action) # calculate q(s,a)
            if v_s > v_s_max:
                v_s_max = v_s
                action_max = action
        return action_max

    def policy_iteration(self, debug=False):
        """
        Using policy evaluation and policy iteration as subroutines, 
        Calculates optimal value function and optimal policy.
        This is not a pure function.
        """
        is_policy_stable = False
        iter = 0
        while (not is_policy_stable):
            self.evaluate_policy()
            is_policy_stable = self.improve_policy()
            debug and print(self.value_fn)
            debug and print(self.policy)
            iter+=1
        print(f"Policy Iteration converged at step {iter}.")


    def value_iteration(self, debug=False):
        """
        Using policy evaluation for state and policy improvement as subroutines,
        Calculates optimal value function first and extracts optimal policy from it.
        This is not a pure function.
        """
        # find optimal value fn
        delta = self.theta + 1
        iter = 0
        while delta > self.theta:
            delta = 0
            debug and print(self.value_fn)
            for s in range(self.mdp.num_states):
                value_s_old = self.value_fn[s]
                for a in range(self.mdp.num_actions):
                    v_s_a = self.evaluate_policy_for_state(s, a)
                    self.value_fn[s] = max(v_s_a, self.value_fn[s])
                delta = max(delta, np.abs(value_s_old - self.value_fn[s]))
            iter+=1
        # extract optimal policy
        print(f"Value Iteration converged at step {iter}.")
        self.improve_policy(debug=debug)

                

    def print_agent_info(self):
        for s in range(self.mdp.num_states):
            a = self.get_action(s)
            v = self.value_fn[s]
            print(f"State {s}, policy(s): {a}, value(s): {v}")

if __name__ == '__main__':
    from MarkovDecisionProcess import MarkovDecisionProcess as MDP
    import gym
    env = gym.make('FrozenLake-v1', is_slippery=False)
    mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)
    agent = Agent(mdp, 1.0)
    agent.policy[0] = [1,0,0,0]
    assert agent.get_action(0) == 0
    agent.policy[14] = [0,0,1,0]
    assert agent.get_action(14) == 2
    agent.policy[2] = [0.5, 0.5, 0, 0]
    assert agent.get_action(2) in [0,1]