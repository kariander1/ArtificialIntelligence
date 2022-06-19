import math
from copy import deepcopy
import numpy as np

actions_mapping = ['UP', 'DOWN', 'RIGHT', 'LEFT']


def is_float(st):
    try:
        return float(st)
    except ValueError:
        return None


def _calc_prob_sum(mdp, U, action_wanted, S):
    prob_sum = 0
    for i_action, action_prob in enumerate(mdp.transition_function[action_wanted]):
        new_state = mdp.step(S, actions_mapping[i_action])
        prob_sum += action_prob * U[new_state[0]][new_state[1]]
    return prob_sum


def _evaluate_state_iteration(mdp, U, S,eps = 0.0):
    reward = is_float(mdp.board[S[0]][S[1]])  # Reward for each cell R(s)
    if reward is None:
        return None, None, None
    prob_sums = {}
    max_actions = []
    max_sum = 0
    if S not in mdp.terminal_states:
        for action_wanted in mdp.actions.keys():
            prob_sums[action_wanted] = _calc_prob_sum(mdp, U, action_wanted, S)
            # if prob_sum > max_sum:
            #     max_sum = prob_sum
            #     max_actions = [action_wanted]
            # elif prob_sum == max_sum:
            #     max_actions += [action_wanted]
        max_sum = max(prob_sums.values())
        max_actions = [a for a in mdp.actions.keys() if prob_sums[a]>=max_sum-eps]
    # Handle cases of terminal states, assign default first actions (won't be used)
    if len(max_actions) == 0:
        max_actions = [actions_mapping[0]]
    return max_sum, max_actions, reward


def _value_iteration_policy(mdp, U):
    delta = 0
    U_temp = deepcopy(U)
    policy = deepcopy(U)
    for i in range(len(mdp.board)):
        for j in range(len(mdp.board[i])):
            S = (i, j)
            max_sum, max_action, Reward = _evaluate_state_iteration(mdp, U, S)
            if max_sum is None:
                continue
            policy[i][j] = max_action[0]
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
        U_temp, delta, _ = _value_iteration_policy(mdp, U)

    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    _, _, policy = _value_iteration_policy(mdp, U)
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    actions_mapping = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    n = mdp.num_col * mdp.num_row
    I = np.eye(n)
    P = np.zeros((n, n))
    R = np.zeros((n, 1))
    for i in range(len(mdp.board)):
        for j in range(len(mdp.board[i])):
            Reward = is_float(mdp.board[i][j])  # Reward for each cell R(s)
            if Reward is None:
                continue
            R[i + j * mdp.num_row] = Reward
            if (i, j) in mdp.terminal_states:
                continue
            prob_vec = np.zeros((1, n))
            chosen_action = policy[i][j]
            for i_action, action_prob in enumerate(mdp.transition_function[chosen_action]):
                new_state = mdp.step((i, j), actions_mapping[i_action])
                col_index = new_state[0] + new_state[1] * mdp.num_row
                prob_vec[0, col_index] += action_prob

            P[i + j * mdp.num_row, :] = prob_vec

    U = np.linalg.inv(I - mdp.gamma * P) @ R
    U = U.reshape((mdp.num_col, mdp.num_row)).T.tolist()
    return U

    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    unchanged = False
    policy = deepcopy(policy_init)
    while not unchanged:
        U = policy_evaluation(mdp, policy)
        unchanged = True
        for i in range(len(mdp.board)):
            for j in range(len(mdp.board[i])):
                S = (i, j)
                max_sum, max_action, _ = _evaluate_state_iteration(mdp, U, S)
                if max_sum is None or S in mdp.terminal_states:
                    continue
                current_sum = _calc_prob_sum(mdp, U, policy[i][j], S)
                if max_sum > current_sum:
                    policy[i][j] = max_action[0]
                    unchanged = False
    return policy
    # ========================
