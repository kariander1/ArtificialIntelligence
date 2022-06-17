""" You can import what ever you want """
import pygame
from copy import deepcopy
from value_and_policy_iteration import _evaluate_state_iteration_

def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belmann equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(U)
    for i in range(len(mdp.board)):
        for j in range(len(mdp.board[i])):
            S = (i, j)
            max_sum, max_actions, Reward = _evaluate_state_iteration_(mdp, U, S)
            if max_sum is None:
                continue
            policy[i][j] = max_actions


    return policy
    # ========================


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
