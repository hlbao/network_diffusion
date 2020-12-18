#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random as rnd
import csv

# Runtime parameters
num_agent = 10000     # Number of agents
average_degree = 8    # Average degree of the social network
num_ens = 100         # Number of simulation run
num_season = 500      # Maximum season

S = 0
I = 1
R = 2
V = 3
NV = 4

class Agent:
    """Definition of agent"""

    def __init__(self, id):
        self.id = id
        self.state = S
        self.next_state = None
        self.num_I = 0
        self.transition_probability = 0
        self.strategy = NV
        self.next_strategy = None
        self.point = 0
        self.neighbors = []

class Society(Agent):
    """Functions to generate population and check fractions"""

    def __init__(self, population_size, average_degree):
        rearange_edges = int(average_degree*0.5)
        self.size = population_size
        self.topology = nx.barabasi_albert_graph(population_size, rearange_edges)

    def network(self, agents):
        """Link all agents based on the underlying network topology"""

        for focal in agents:
            neighbors_id = list(self.topology[focal.id])

            for nb_id in neighbors_id:
                focal.neighbors.append(agents[nb_id])

        return agents
    
 
    def generate_agents(self):
        """Generate a list of agents connected with network"""

        agents = [Agent(id) for id in range(self.size)]
        connected_agents = self.network(agents)

        return connected_agents

    def count_fraction(self, agents):
        """Calculate the ratio of suseptible, infected recovered people and vaccinators"""

        Fs = len([agent for agent in agents if agent.state == S])/self.size
        Fi = len([agent for agent in agents if agent.state == I])/self.size
        Fr = len([agent for agent in agents if agent.state == R])/self.size
        Fv = len([agent for agent in agents if agent.strategy == V])/self.size

        return Fs, Fi, Fr, Fv

    def count_num_I(self, agents):
        """Count the number of I agents"""

        num_I = len([agent for agent in agents if agent.state == I])

        return num_I

    def count_SAP(self, agents):
        """Count Social Average Payoff"""

        SAP = np.mean([agent.point for agent in agents])

        return SAP

class Decision:
    """Functions related to decision making process"""

    def __init__(self, Cr):
        self.Cr = Cr
        self.kappa = 0.1  # Thermal coefficient of PW-Fermi Comparison

    def choose_initial_vaccinators(num_agent):
        """Return the ID list of initial vaccinators"""

        init_vaccinators_id = rnd.sample([id for id in range(num_agent)], k = int(num_agent/2))

        return init_vaccinators_id

    def init_strategy(self, agents, init_vaccinators_id):
        """Set initial strategy when varying Cr"""

        for focal in agents:
            if focal.id in init_vaccinators_id:
                focal.strategy = V
            else:
                focal.strategy = NV

        return agents

    def payoff(self, agents):
        """Count payoff"""

        for focal in agents:
            if focal.strategy == V and focal.state == S:
                focal.point = -self.Cr

            if focal.strategy == V and focal.state == R:
                focal.point = -self.Cr-1.0

            if focal.strategy == NV and focal.state == S:
                focal.point = 0.0

            if focal.strategy == NV and focal.state == R:
                focal.point = -1.0

        return agents

    def PW_Fermi(self, agents):
        """Decide the strategy for next season based on Pairwise Fermi comparison"""

        for focal in agents:
            opp = rnd.choice(focal.neighbors)
            if rnd.random() <= 1/(1 + np.exp(focal.point - opp.point)/self.kappa):
                focal.next_strategy = opp.strategy  # The rest of the agents keep their previous next_strategy
            else:
                focal.next_strategy = focal.strategy

        return agents

    def update_strategy(self, agents):
        """Update the strategy of all agents synchronously"""

        agents = self.PW_Fermi(self.payoff(agents))

        for focal in agents:
            focal.strategy = focal.next_strategy

        return agents

