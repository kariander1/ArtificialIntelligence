""" You can import what ever you want """
import argparse

import numpy as np

from main import is_valid_file
from copy import deepcopy
from mdp import MDP
from termcolor import colored
from value_and_policy_iteration import is_float, _evaluate_state_iteration, value_iteration, get_policy


def get_all_policies(mdp, U, prev_policies=None, eps=1e-4):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belmann equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    n_policies = 1
    unchanged = True
    policies = deepcopy(U)
    for i in range(len(mdp.board)):
        for j in range(len(mdp.board[i])):
            S = (i, j)
            max_sum, max_actions, Reward = _evaluate_state_iteration(mdp, U, S, eps)
            if max_sum is None:
                continue
            policies[i][j] = max_actions
            if prev_policies:
                prev_actions = prev_policies[i][j]
                if (len(max_actions) != len(prev_actions)) or max_actions != prev_actions:
                    unchanged = False
            n_policies *= len(max_actions)

    # print_policies(mdp, policies)
    return n_policies, policies, unchanged
    # ========================


def print_policies(mdp, policies):
    res = ""
    cell_row_size = 3
    cell_col_size = 7
    center_pos = (cell_row_size // 2, cell_col_size // 2)
    char_mapping = {
        "UP": "↑",
        "DOWN": "↓",
        "RIGHT": "→",
        "LEFT": "←",
        "wall": "█",
        "T": "█"
    }
    char_color = {
        "UP": 'yellow',
        "DOWN": 'yellow',
        "RIGHT": 'yellow',
        "LEFT": 'yellow',
        "wall": 'blue',
        "WALL": 'blue',

    }
    char_rel_pos = {
        "UP": [(0, 3)],
        "DOWN": [(2, 3)],
        "RIGHT": [(1, 5)],
        "LEFT": [(1, 1)],
        "wall": [(i, j) for i in range(cell_row_size) for j in range(cell_col_size)],
        "T": [(i, j) for i in range(cell_row_size) for j in range(cell_col_size)]
    }
    table_width = 1 + mdp.num_col * (cell_col_size + 1) + 1

    for r in range(mdp.num_row):
        if r == 0:
            res += "╔"
            res += "╦".join(["═" * cell_col_size] * mdp.num_col)
            res += "╗"
        else:
            res += "╠"
            res += "╬".join(["═" * cell_col_size] * mdp.num_col)
            res += "╣"
        res += '\n'
        empty_row = "║"
        empty_row += "║".join([" " * cell_col_size] * mdp.num_col)
        empty_row += "║"
        res += "\n".join([empty_row] * cell_row_size)
        res += '\n'
        if r == mdp.num_row - 1:
            res += "╚"
            res += "╩".join(["═" * cell_col_size] * mdp.num_col)
            res += "╝"
    res = list(res)
    for i in range(len(policies)):
        for j in range(len(policies[i])):
            actions = policies[i][j]
            if actions == 0:
                actions = [mdp.board[i][j].lower(), "WALL"]

            if (i, j) in mdp.terminal_states:
                actions = ["T", mdp.board[i][j]]

            for action in actions:
                chars_to_place = char_mapping.get(action, action)
                char_poses = char_rel_pos.get(action, [center_pos])
                color = char_color.get(action, 'red')
                for char_pos in char_poses:
                    for char_i, char_to_place in enumerate(chars_to_place):
                        row_in_table = 1 + i * (cell_row_size + 1) + char_pos[0]
                        col_in_table = 1 + j * (cell_col_size + 1) + char_pos[1]
                        ind_in_table = row_in_table * table_width + col_in_table
                        ind_in_table -= ((len(chars_to_place) - 1) // 2 - char_i)
                        res[ind_in_table] = colored(char_to_place, color)
    print("".join(res))


def get_policy_for_different_rewards(mdp,r_res = 0.005):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displays the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    initial_u = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]

    print("\nInitial utility:")
    mdp.print_utility(initial_u)

    joint_rewards = [is_float(x) for x in sum([x for x in mdp.board], []) if is_float(x)]
    min_r = min(joint_rewards) - max(joint_rewards)
    max_r = 0 if mdp.gamma == 1 else max(joint_rewards)


    np_mdp = np.array(mdp.board)
    cell_mask = calc_cell_mask(mdp)

    policies_list = []
    prev_policies = None

    for r in np.arange(min_r, max_r + r_res, r_res):
        reward = round(r.item(), 5)
        current_mdp = deepcopy(mdp)
        current_mdp_np = np.copy(np_mdp)
        current_mdp_np[cell_mask] = str(reward)
        current_mdp.board = current_mdp_np.tolist()

        U = value_iteration(current_mdp, U_init=initial_u)
        _, policies, unchanged = get_all_policies(mdp, U, prev_policies)
        if not unchanged or not policies_list:
            policies_list += [[reward+r_res, policies]]
        prev_policies = policies

    for i, (reward, policies) in enumerate(policies_list):
        print_policies(mdp, policies)
        range_str = ""
        if i > 0:
            op_sign = "<="
            range_str += "{:.3f}".format(reward) + op_sign
        range_str += "R(s)"
        if i < len(policies_list) - 1:
            range_str += "<{:.3f}".format(policies_list[i + 1][0])
        print(range_str)
    # ========================


def calc_cell_mask(mdp):
    np_mdp = np.array(mdp.board)
    mask = ~np.isnan(np.vectorize(lambda a: is_float(a))(np_mdp))
    terminal_states = np.array(mdp.terminal_states)
    mask[terminal_states[:, 0], terminal_states[:, 1]] = False

    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Board", help="A file that holds the board and the reward for each state")
    parser.add_argument("TerminalStates", help="A file that contains the terminal states in the board")
    parser.add_argument("TransitionFunction", help="A file that contains the transition function")
    args = parser.parse_args()

    board = args.Board
    terminal_states = args.TerminalStates
    transition_function = args.TransitionFunction

    is_valid_file(parser, board)
    is_valid_file(parser, terminal_states)

    board_env = []
    with open(board, 'r') as f:
        for line in f.readlines():
            row = line[:-1].split(',')
            board_env.append(row)

    terminal_states_env = []
    with open(terminal_states, 'r') as f:
        for line in f.readlines():
            row = line[:-1].split(',')
            terminal_states_env.append(tuple(map(int, row)))

    transition_function_env = {}
    with open(transition_function, 'r') as f:
        for line in f.readlines():
            action, prob = line[:-1].split(':')
            prob = prob.split(',')
            transition_function_env[action] = tuple(map(float, prob))

    # initialising the env
    mdp = MDP(board=board_env,
              terminal_states=terminal_states_env,
              transition_function=transition_function_env,
              gamma=0.9)

    get_policy_for_different_rewards(mdp)
