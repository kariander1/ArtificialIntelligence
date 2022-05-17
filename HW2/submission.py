import math
import time
from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv
from TaxiEnv import manhattan_distance as md
import random

dt_epsilon = 0.05
max_utility = 10000

def chosen_utility(env: TaxiEnv, taxi_id: int):
    assert env.done()
    taxi = env.get_taxi(taxi_id)
    other_taxi = env.get_taxi(1 - taxi_id)
    taxi_wins = taxi.cash > other_taxi.cash
    return max_utility + taxi.cash if taxi_wins else -max_utility - other_taxi.cash


def chosen_heuristic(env: TaxiEnv, taxi_id: int):
    taxi = env.taxis[taxi_id]
    other_taxi = env.taxis[1 - taxi_id]

    K = taxi.passenger is not None
    D = taxi.cash - other_taxi.cash

    # A is best reward from the trip with passenger
    A = -math.inf
    p = None
    for passenger in env.passengers:
        dist_to_passenger = md(taxi.position, passenger.position)
        dist_passenger_to_dest = md(passenger.position, passenger.destination)
        A_temp = dist_passenger_to_dest - dist_to_passenger
        if A_temp >= A and taxi.fuel >= (dist_to_passenger + dist_passenger_to_dest):
            A = A_temp
            p = passenger.position

    # C is the distance from taxi to its destination
    B = 0
    C = math.inf
    if K:
        B = md(taxi.passenger.position, taxi.passenger.destination)
        C = md(taxi.position, taxi.passenger.destination)
    elif p:
        C = md(taxi.position, p)
    else:
        for gas_station in env.gas_stations:
            C = min(C, md(taxi.position, gas_station.position))

    H = (B - C + D)
    return H


class _MiniMaxNode:
    def __init__(self, val, op):
        self.val = val
        self.op = op

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return self == other or self < other

    def __lt__(self, other):
        operand = other
        if isinstance(other, type(self)):
            operand = operand.val
        return self.val < operand

    def __eq__(self, other):
        operand = other
        if isinstance(other, type(self)):
            operand = operand.val
        return self.val == operand

    def __neg__(self):
        return _MiniMaxNode(-self.val, self.op)

    def __repr__(self):
        return "val={} op={}".format(self.val, self.op)


def negamax(agent, env: TaxiEnv, agent_id, depth, use_prune=False, alpha=-math.inf, beta=math.inf):
    # Update min depth explored:
    agent.min_depth_remaining = min(agent.min_depth_remaining, depth)

    # If evaluated env is done, return utility
    minimax_mod = 1 if agent_id == agent.turn_id else -1
    if env.done():
        return chosen_utility(env, agent_id) * minimax_mod
    # If reached min depth or time finished - return heuristic
    if depth == 0 or agent.remaining_time() <= 0:
        return agent.heuristic(env, agent_id) * minimax_mod

    # Fetch legal operators in current state
    operators = env.get_legal_operators(agent_id)
    # Clone env for each operator
    children = [env.clone() for _ in operators]

    # Initialize minmax value and chose operation
    minmax_val = -math.inf
    chosen_op = None
    for child, op in zip(children, operators):
        # Apply the operator to the environment
        child.apply_operator(agent_id, op)
        # Fetch current negamax value
        current_val = -negamax(agent, child, 1 - agent_id, depth - 1, use_prune, -beta, -alpha)
        # If current value is better, update minmax value and chosen op
        if current_val > minmax_val:
            minmax_val = current_val
            chosen_op = op
        # Update alpha value (it will be beta on each consecutive call)
        alpha = max(alpha, minmax_val)
        if use_prune and alpha >= beta:
            # If we are pruning, return alpha (the node won't be selected)
            return alpha
    # If we are returning at the root return also the chose operation
    return _MiniMaxNode(minmax_val, chosen_op) if depth == agent.initial_depth else minmax_val


class AgentGreedyImproved(AgentGreedy):

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        return AgentGreedy.run_step(self, env, agent_id, time_limit)

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return chosen_heuristic(env, taxi_id)


class AgentMinimax(Agent):
    def remaining_time(self):
        return self.time_limit - (time.time() - self.start_time) - dt_epsilon

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.turn_id = agent_id
        self.min_depth_remaining = 0
        depth = 1
        minmax_val = -math.inf
        # Continue iterative deepening as long as time remains and we haven't reached
        # the bottom of the game tree
        while self.remaining_time() > 0 and self.min_depth_remaining == 0:
            self.min_depth_remaining = math.inf
            self.initial_depth = depth
            minmax_val = max(minmax_val, negamax(self, env, agent_id, depth))
            depth = depth + 1
        # Patch for handling only root game tree
        if not isinstance(minmax_val,_MiniMaxNode):
            minmax_val=_MiniMaxNode(0,env.get_legal_operators(agent_id)[0])
        return minmax_val.op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return chosen_heuristic(env, taxi_id)


class AgentAlphaBeta(Agent):
    def remaining_time(self):
        return self.time_limit - (time.time() - self.start_time) - dt_epsilon

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.turn_id = agent_id
        self.min_depth_remaining = 0
        depth = 1
        minmax_val = -math.inf
        # Continue iterative deepening as long as time remains and we haven't reached
        # the bottom of the game tree
        while self.remaining_time() > 0 and self.min_depth_remaining == 0:
            self.min_depth_remaining = math.inf
            self.initial_depth = depth
            minmax_val = max(minmax_val, negamax(self, env, agent_id, depth, True))
            depth = depth + 1
        # Patch for handling only root game tree
        if not isinstance(minmax_val,_MiniMaxNode):
            minmax_val=_MiniMaxNode(0,env.get_legal_operators(agent_id)[0])
        return minmax_val.op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return chosen_heuristic(env, taxi_id)


class AgentExpectimax(Agent):
    move_ops = ['move north', 'move south', 'move east', 'move west']

    def _getWeight(self, op):
        # Return weight 1 for all move operations
        return 1 if op in self.move_ops else 2

    def _expectiMax(self, env: TaxiEnv, agent_id, depth):
        # Update min depth explored:
        self.min_depth_remaining = min(self.min_depth_remaining, depth)

        # If evaluated env is done, return utility
        if env.done():
            return chosen_utility(env, agent_id)

        # If reached min depth or time finished - return heuristic
        if depth == 0 or self.remaining_time() <= 0:
            return self.heuristic(env, agent_id)

        # Fetch legal operators in current state
        operators = env.get_legal_operators(agent_id)
        # Clone env for each operator
        children = [env.clone() for _ in operators]
        # Calc weights of all operations
        weights = [self._getWeight(op) for op in operators]
        # Initialize mean and minmax val and chosen operation
        expectation = 0
        minmax_val = -math.inf
        chosen_op = None

        for child, op, weight in zip(children, operators, weights):
            # Apply operator on each child
            child.apply_operator(agent_id, op)
            # cal childs expectiMax value
            current_val = self._expectiMax(child, 1 - agent_id, depth - 1)
            # If value is better, update minimax value and chosen op
            if current_val > minmax_val:
                minmax_val = current_val
                chosen_op = op
            # Calculate contribution of operation to expectation value
            expectation += (self._getWeight(op) / sum(weights)) * current_val

        # Chose minmax val to have the chosen operation of it is the root
        minmax_val = _MiniMaxNode(minmax_val, chosen_op) if depth == self.initial_depth else minmax_val
        # return minmax val if it is a minmax node, otherwise an expectimax
        return minmax_val if agent_id == self.turn_id else expectation

    def remaining_time(self):
        return self.time_limit - (time.time() - self.start_time) - dt_epsilon

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.turn_id = agent_id
        self.min_depth_remaining = 0
        depth = 1
        minmax_val = -math.inf
        # Continue iterative deepening as long as time remains and we haven't reached
        # the bottom of the game tree
        while self.remaining_time() > 0 and self.min_depth_remaining == 0:
            self.min_depth_remaining = math.inf
            self.initial_depth = depth
            minmax_val = max(minmax_val, self._expectiMax(env, agent_id, depth))
            depth = depth + 1
        # Patch for handling only root game tree
        if not isinstance(minmax_val,_MiniMaxNode):
            minmax_val=_MiniMaxNode(0,env.get_legal_operators(agent_id)[0])
        return minmax_val.op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return chosen_heuristic(env, taxi_id)
