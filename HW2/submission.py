from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random



class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        return AgentGreedy.run_step(self, env, agent_id, time_limit)


    def heuristic(self, env: TaxiEnv, taxi_id: int):

        I1 = manhattan_distance(env.taxis[taxi_id].position,env.gas_stations[0].position)
        I2 = manhattan_distance(env.taxis[taxi_id].position,env.gas_stations[1].position)
        I = (I1< env.taxis[taxi_id].fuel) or (I2<env.taxis[taxi_id].fuel)
        K = env.taxis[taxi_id].passenger != None
        if not K:
            dist_to_passenger1 = manhattan_distance(env.taxis[taxi_id].position,env.passengers[0].position)
            dist_to_passenger2 = manhattan_distance(env.taxis[taxi_id].position, env.passengers[1].position)
            A1 = manhattan_distance(env.passengers[0].position,env.passengers[0].destination)
            A2 = manhattan_distance(env.passengers[1].position,env.passengers[1].destination)
            A1 = A1 - dist_to_passenger1
            A2 = A2 - dist_to_passenger2
            A = max(A1,A2)*I
        else: A = 0
        B = -min(I1,I2)*(1-I)
        if K:
            C = manhattan_distance(env.taxis[taxi_id].position, env.taxis[taxi_id].passenger.destination)
            C = (8-C) * I
        else: C = 0
        H = ((A+B+C)/16)
        print("A={} B={} C={} I={} K={} H={}".format(A,B,C,I,K,H))
        return H


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
