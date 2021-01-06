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

class Epidemics:
    """Functions related to SIR dynamics"""

    def __init__(self, effectiveness):
        self.beta = 2                          # Infection probability
        self.mu = 0.33                         # Recovery probability
        self.num_initI = 5                     # Number of initially infected people
        self.effectiveness = effectiveness     # Effectiveness of Vaccination
        self.total_transition_probability = 0  # Sum of the transition probability over all agents
        self.accum_days = 0                    # Accumulative day calculated by Gillespie algorithm(count_day method)

    def reset_day(self):
        """Initialize accum_day at the beginning of each season"""

        self.accum_days = 0

    def init_state(self, agents):
        """Set initial state for all agents & call self.transition_probability"""

        NV_id = [agent.id for agent in agents if agent.strategy == NV]
        init_infectors_id = rnd.sample(NV_id, k = self.num_initI)  # Initial infected people should be selected from non-vaccinators

        # Set initial state
        for focal in agents:
            if focal.id in init_infectors_id:
                focal.state = I
            else:
                focal.state = S

        # Count the number of neighboring I agents
        for focal in agents:
            focal.num_I = len([neighbor for neighbor in focal.neighbors if neighbor.state == I])

        # Set transition probability for all agents
        agents = self.transition_probability(agents)

        return agents

    def transition_probability(self, agents):
        """Set transition probabilities for all agents and take the sum of them"""

        self.total_transition_probability = 0

        for focal in agents:
            if focal.state == S:
                focal.transition_probability = self.beta*focal.num_I

            if focal.state == I:
                focal.transition_probability = self.mu

            if focal.state == R:
                focal.transition_probability = 0

            self.total_transition_probability += focal.transition_probability

        return agents

    def state_change(self, agents):
        """Update SIR state based on Gillespie Algorithm"""

        rand_num = rnd.random()
        probability = 0

        for focal in [agent for agent in agents if agent.state != R and agent.transition_probability != 0]:
            probability += focal.transition_probability/self.total_transition_probability

            if rand_num <= probability:  # This random number is fixed for all agents
                previous_transition_prob = focal.transition_probability

                # For Failed SV agent
                if focal.state == S and focal.strategy == V:
                    if rnd.random() > self.effectiveness:
                        focal.next_state = I
                        focal.transition_probability = self.mu

                        for neighbor in focal.neighbors:
                            neighbor.num_I += 1
                            self.update_total_transition_probability(neighbor)

                    else:
                        # Not necessary but for safety
                        focal.next_state = S
                        focal.transition_probability = self.beta*focal.num_I

                # For NV agent
                if focal.state == S and focal.strategy == NV:
                    focal.next_state = I
                    focal.transition_probability = self.mu

                    for neighbor in focal.neighbors:
                        neighbor.num_I += 1
                        self.update_total_transition_probability(neighbor)

                # For I agent
                if focal.state == I:
                    focal.next_state = R
                    focal.transition_probability = 0

                    for neighbor in focal.neighbors:
                        neighbor.num_I -= 1
                        self.update_total_transition_probability(neighbor)

                focal.state = focal.next_state
                self.total_transition_probability -= previous_transition_prob
                self.total_transition_probability += focal.transition_probability

                break  # only one agent should be picked up for one call of this method

        return agents

    def update_total_transition_probability(self, neighbor):
        """Update total transition probability if the given neighbor is S state connected with newly generated I agent"""

        if neighbor.state == S:
            self.total_transition_probability -= neighbor.transition_probability
            neighbor.transition_probability = self.beta*neighbor.num_I
            self.total_transition_probability += neighbor.transition_probability

    def count_day(self):
        """Calculate accumulative days"""

        delta_day = np.log(1/rnd.random())/self.total_transition_probability
        self.accum_days += delta_day

        return self.accum_days
