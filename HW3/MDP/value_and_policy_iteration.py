import math
from copy import deepcopy
import numpy as np


def is_float(st):
    try:
        return float(st)
    except ValueError:
        return None


def _value_interation_policy(mdp, U):
    actions_mapping = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    delta = 0
    U_temp = deepcopy(U)
    policy = deepcopy(U)
    for i in range(len(mdp.board)):
        for j in range(len(mdp.board[i])):
            Reward = is_float(mdp.board[i][j])  # Reward for each cell R(s)
            if Reward is None:
                continue
            max_action = actions_mapping[0]
            max_sum = 0
            if (i, j) not in mdp.terminal_states:
                for action_wanted in mdp.actions.keys():
                    prob_sum = 0
                    for i_action, action_prob in enumerate(mdp.transition_function[action_wanted]):
                        new_state = mdp.step((i, j), actions_mapping[i_action])
                        prob_sum += action_prob * U[new_state[0]][new_state[1]]

                    if prob_sum > max_sum:
                        max_sum = prob_sum
                        max_action = action_wanted

            policy[i][j] = max_action
            U_temp[i][j] = Reward + mdp.gamma * max_sum
            delta = max(delta, abs(U_temp[i][j] - U[i][j]))
    return U_temp, delta, policy


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U_temp = deepcopy(U_init)
    U = deepcopy(U_init)
    delta = math.inf

    while mdp.gamma != 0 and (delta >= epsilon * (1 - mdp.gamma) / mdp.gamma) and delta > 0:
        U = deepcopy(U_temp)
        U_temp, delta, _ = _value_interation_policy(mdp, U)

    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    _, _, policy = _value_interation_policy(mdp, U)
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
