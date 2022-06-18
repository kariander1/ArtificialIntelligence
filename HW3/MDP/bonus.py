""" You can import what ever you want """
import pygame
from copy import deepcopy

from termcolor import colored

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

def print_policies(mdp, policies):
    res = ""
    cell_row_size = 3
    cell_col_size = 7
    char_mapping = {
        "UP": "▲",
        "DOWN": "▼",
        "RIGHT": "►",
        "LEFT": "◄",
        "WALL": colored("█",'blue')
    }
    char_rel_pos = {
        "UP": [(0,3)],
        "DOWN": [(2,3)],
        "RIGHT": [(1,5)],
        "LEFT": [(1,1)],
        "WALL": [(i, j) for i in range(cell_row_size) for j in range(cell_col_size)]
    }
    table_width = 1 + mdp.num_col * (cell_col_size + 1) + 1

    for r in range(mdp.num_row):
        if r== 0:
            res += "╔"
            res += "╦".join(["═"* cell_col_size] * mdp.num_col )
            res += "╗"
        else:
            res += "╠"
            res += "╬".join(["═"* cell_col_size]*mdp.num_col)
            res += "╣"
        res += '\n'
        empty_row = "║"
        empty_row += "║".join([" "*cell_col_size]*mdp.num_col)
        empty_row += "║"
        res+= "\n".join([empty_row]*cell_row_size)
        res+= '\n'
        #
        # for c in range(mdp.num_col):
        #     if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
        #         val = mdp.board[r][c]
        #     else:
        #         val = str(policies[r][c])
        #     if (r, c) in mdp.terminal_states:
        #         res += " " + colored(val[:5].ljust(5), 'red') + " |"  # format
        #     elif mdp.board[r][c] == 'WALL':
        #         res += " " + colored(val[:5].ljust(5), 'blue') + " |"  # format
        #     else:
        #         res += " " + val[:5].ljust(5) + " |"  # format
        # res += "\n"
        if r==mdp.num_row-1:
            res += "╚"
            res += "╩".join(["═"*cell_col_size] * mdp.num_col)
            res += "╝"
    res = list(res)
    for i in range(len(policies)):
        for j in range(len(policies[i])):
            actions = policies[i][j]
            if actions == 0:
                actions = [mdp.board[i][j]]
            terminal_state = False
            if (i, j) in mdp.terminal_states:
                continue
                terminal_state = True
            for action in actions:
                char_to_place = char_mapping[action]
                char_poses = char_rel_pos[action]
                for char_pos in char_poses:
                    row_in_table = 1 + i * (cell_row_size + 1) + char_pos[0]
                    col_in_table = 1 + j * (cell_col_size + 1) + char_pos[1]
                    ind_in_table = row_in_table * table_width + col_in_table
                    res[ind_in_table] = char_to_place
    print("".join(res))


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
