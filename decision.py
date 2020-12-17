
import numpy as np
import random as rnd

def choose_initial_vaccinators(num_agent):
    """Return agent id list which used for one episode"""

    initial_vaccinators_id = rnd.sample(range(num_agent), k = num_agent//2)

    return initial_vaccinators_id

def initialize_strategy(agents, initial_vaccinators_id):
    """Initialize the strategy of all agent with the given initial_vaccinators_id"""

    for agent_id, agent in enumerate(agents):
       if agent_id in initial_vaccinators_id:
           agent.strategy = 'V'
       else:
           agent.strategy = 'NV'

def PW_Fermi(agents):
    """Determine next_strategy of all agents for the next season"""

    kappa = 0.1  # Thermal coefficient

    # Randomely select one neighboring agent as opponent and determine whether or not copying his strategy
    for focal in agents:
        opponent_id = rnd.choice(focal.neighbors_id)
        opponent = agents[opponent_id]
        if opponent.strategy != focal.strategy and rnd.random() <= 1/(1+np.exp((focal.point-opponent.point)/kappa)):
            focal.next_strategy = opponent.strategy
        else:
            focal.next_strategy = focal.strategy

    # Update strategy
    for agent in agents:
        agent.strategy = agent.next_strategy
