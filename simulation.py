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
