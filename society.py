import numpy as np
import networkx as nx

class Agent:

    def __init__(self):
        """
        state = ['S', 'IM', 'I', 'R'] (S: Susceptible, IM: Immuned, I: Infectious, R: Recovered)
        strategy = ['V', 'NV']        (V: Vaccinator, NV: Non-vaccinator)
        """
        self.state = 'S'
        self.next_state = 'S'
        self.strategy = 'V'
        self.next_strategy = 'NV'
        self.point = 0
        self.neighbors_id = None

def generate_agents(num_agent, average_degree):
    network = nx.barabasi_albert_graph(num_agent, average_degree//2)
    agents = [Agent() for agent_id in range(num_agent)]

    for agent_id, agent in enumerate(agents):
        agent.neighbors_id = list(network[agent_id])

    return agents

def count_state_fraction(agents):
    """Count the fraction of S/IM/I/R state agents"""

    fs  = len([agent for agent in agents if agent.state =='S'])/len(agents)
    fim = len([agent for agent in agents if agent.state =='IM'])/len(agents)
    fi  = len([agent for agent in agents if agent.state =='I'])/len(agents)
    fr  = 1 - fs - fim - fi

    return fs, fim, fi, fr

def count_num_i(agents):
    """Count the number of infected agents"""

    num_i = len([agent for agent in agents if agent.state == 'I'])

    return num_i

def count_strategy_fraction(agents):
    """Count the fraction of vaccinators"""

    fv = len([agent for agent in agents if agent.strategy == 'V'])/len(agents)

    return fv

def count_num_nv(agents):
    """Count the number of non-vaccinators"""

    num_nv = len([agent for agent in agents if agent.strategy == 'NV'])

    return num_nv

def count_sap(agents):
    sap = np.mean([agent.point for agent in agents])

    return sap
