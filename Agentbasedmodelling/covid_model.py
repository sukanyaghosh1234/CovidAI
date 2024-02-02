import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.visualization.UserParam import UserSettableParameter
import scipy.stats as ss
from enum import Enum
import math
import random

# Simulation model parameters
# -------------------------------------------------------------------------------------------
model_params = {

    'no_agents': UserSettableParameter('number', 'Number of agents', 41000, 5, 50000, 5),
    'width': 141,
    'height': 141,
    'init_infected': UserSettableParameter('slider', 'Initial infected people(%)', 0.000144, 0, 0.000001, 0.001),
    'infection_period': 14,
    'immunity_period': 60,
    'mortality_rate': UserSettableParameter('slider', 'Mortality rate', 0.01, 0, 1, 0.01),
    'perc_health_worker': 0.0138,
    'lockdown_status': 'Complete Lockdown',
    'protective_measures':'Both',
    'household_size': UserSettableParameter('slider', 'Household size', 2.1, 1, 6, 0.1),
    'isolation_capacity' : 500 #assumption

    # 'lockdown_status': UserSettableParameter('choice', 'Severity of Lockdown',
    #                                          value='No Lockdown',
    #                                          choices=['No Lockdown', 'Partial Lockdown', 'Complete Lockdown']),
    # 'lockdown_status': UserSettableParameter('choice', 'Severity of Lockdown',
    #                                          value='Complete Lockdown', choices=['No Lockdown', 'Partial Lockdown', 'Complete Lockdown']),
    # 'protective_measures': UserSettableParameter('choice', 'Protective Measures',
    #                                          value='Both', choices=['No Measures', 'Mask Mandatory', 'Social Distancing', 'Both']),
    
}

class AgentStatus(Enum):
    susceptible = 'Susceptible'
    dead = 'Dead'
    infected = 'Infected'
    recovered = 'Recovered'
    isolated = 'Isolated'

class LockdownStatus(Enum):
    no_lockdown = 'No Lockdown'
    partial = 'Partial Lockdown'
    complete = 'Complete Lockdown'

class ProtectiveMeasures(Enum):
    masks_mandatory = 'Mask Mandatory'
    social_distancing = 'Social Distancing'
    both = 'Both'
    no_measure = 'No Measures'

class Agent(Agent):
    """Agents in the CovidModel class below"""

    def __init__(self, unique_id, model, lockdown_status, infection_rate, temperature,blood_oxygen_level):
        super().__init__(unique_id, model)
        self.model = model
        self.health_worker = False
        self.recovery_countdown = 0
        self.immunity_countdown = 0
        self.infection_day = 0
        self.temperature = temperature
        self.blood_oxygen_level = blood_oxygen_level
        self.lockdown_status: LockdownStatus = lockdown_status
        self.infection_rate = infection_rate
        self.status = AgentStatus.infected if bool(ss.bernoulli.rvs(self.model.init_infected)) else AgentStatus.susceptible
        self.health_worker = bool(ss.bernoulli.rvs(self.model.perc_health_worker))
        self.recovery_countdown = 0
        self.immunity_countdown = 0
        self.household = None
        self.household_transmission_probability = 0.1

        # Random recovery countdown considers agents got infected at different times
        if self.status == AgentStatus.infected:
            self.recovery_countdown = math.floor(np.random.normal(self.model.infection_period, 2))

    def set_household(self, household):
        self.household = household
        # Set the household transmission probability based on the size of the household
        household_size = len(household.individuals)
        if household_size == 1:
            self.household_transmission_probability = 0.0
        elif household_size == 2:
            self.household_transmission_probability = 0.5
        elif household_size == 3:
            self.household_transmission_probability = 0.75
        else:
            self.household_transmission_probability = 0.9

    def update_infected(self):
        if self.status == AgentStatus.susceptible:
            # Workaround for potential bug
            pos = tuple([int(self.pos[0]), int(self.pos[1])])
            # List of agents in the same grid cell
            #cell_agents = self.model.grid.get_cell_list_contents(pos)
            # Checks if any of the agents in the cell are infected
            #any_infected = any(a.status == AgentStatus.infected for a in cell_agents)

            for house_member in self.household.individuals:
                if house_member.status == AgentStatus.infected:
                    self.status = AgentStatus.infected if bool(ss.bernoulli.rvs(0.63)) else self.status
                    self.infection_day = self.model.schedule.time

            # Check if any neighbors are infected and infect agent
            neighbors = self.model.grid.get_neighbors(pos, moore=True)
            infected_neighbors = any(
                a.status == AgentStatus.infected and 1 < (self.model.schedule.time - a.infection_day) < 11 for a in
                neighbors)

            if infected_neighbors and self.status is not AgentStatus.infected:
                if self.health_worker and self.blood_oxygen_level<90 and self.temperature>100:
                    self.status = AgentStatus.infected if bool(ss.bernoulli.rvs(self.infection_rate)) else self.status
                    self.infection_day = self.model.schedule.time
                else:
                    self.status = AgentStatus.infected if bool(ss.bernoulli.rvs(self.infection_rate)) else self.status
                    self.infection_day = self.model.schedule.time

            # Once infected countdown to recovery begins
            if self.status == AgentStatus.infected:
                self.recovery_countdown = math.floor(np.random.normal(14, 2))

    def update_dead(self):
        # death ration among infected people
        if self.status == AgentStatus.infected and (self.model.schedule.time - self.infection_day) >= 7:
            ##remove from grid
            self.status = AgentStatus.dead if bool(ss.bernoulli.rvs(self.model.mortality_rate)) else self.status

    def check_for_health_worker(self):
        if self.health_worker == False and self.status == AgentStatus.infected :
            neighbors = self.model.grid.get_cell_list_contents(self.pos)
            neighbors.extend(self.model.grid.get_neighbors(self.pos, moore=True, include_center=True))
            health_workers = [agent for agent in neighbors if agent.health_worker]
             #print(health_workers)
            if len(health_workers) > 0:
                if self.model.isolation_capacity > 0:
                    #print("before isolation" + str(self.model.isolation_capacity))
                    self.status = AgentStatus.isolated
                    self.model.isolation_capacity -= 1
                    #print("after isolation" + str(self.model.isolation_capacity))
        if self.health_worker==True and self.status==AgentStatus.infected:
              if self.model.isolation_capacity > 0:
                    #print("before isolation" + str(self.model.isolation_capacity))
                    self.status = AgentStatus.isolated
                    self.model.isolation_capacity -= 1
                    #print("after isolation" + str(self.model.isolation_capacity))

    def update_recovered(self):
        if self.recovery_countdown == 1 and self.status == AgentStatus.isolated:
            #print("before recovry" + str(self.model.isolation_capacity))
            self.status = AgentStatus.recovered
            # After recovery countdown to immunity going away begins
            self.immunity_countdown = self.model.immunity_period
            self.model.isolation_capacity += 1
            #print("after recovery" + str(self.model.isolation_capacity))
        elif self.recovery_countdown == 1:
            self.status = AgentStatus.recovered
            # After recovery countdown to immunity going away begins
            self.immunity_countdown = self.model.immunity_period

        if self.recovery_countdown > 0:
            self.recovery_countdown += -1

    def update_susceptible(self):
        # After immunity wanes away, agent becomes susceptible
        if self.immunity_countdown == 1:
            self.status = AgentStatus.susceptible
        if self.immunity_countdown > 0:
            self.immunity_countdown += -1

    def random_activation(self):
        if bool(ss.bernoulli.rvs(self.model.init_infected)):
            self.status = AgentStatus.infected
            self.recovery_countdown = math.floor(np.random.normal(14, 2))

    def move(self):
        if self.status is not AgentStatus.dead and self.status is not AgentStatus.isolated and not self.health_worker:
            # possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
            # new_position = self.random.choice(possible_steps)
            if self.lockdown_status == LockdownStatus.no_lockdown.value:
                if bool(ss.bernoulli.rvs(0.9)):
                    x = self.random.randrange(self.model.grid.width)
                    y = self.random.randrange(self.model.grid.height)
                    self.model.grid.move_agent(self, (x, y))
            elif self.lockdown_status == LockdownStatus.partial.value:
                # print(f"Agent from pos {self.pos[0]} and {self.pos[1]} with state {self.status}")
                if bool(ss.bernoulli.rvs(0.5)):
                    x = self.random.randrange(self.model.grid.width)
                    y = self.random.randrange(self.model.grid.height)
                    # print(f"Agent from pos {self.pos[0]} and {self.pos[1]} with state {self.status}")
                    self.model.grid.move_agent(self, (x, y))
            elif self.lockdown_status == LockdownStatus.complete.value:
                if bool(ss.bernoulli.rvs(0.1)):
                    # print(f"Only I am moving")
                    x = self.random.randrange(self.model.grid.width)
                    y = self.random.randrange(self.model.grid.height)
                    self.model.grid.move_agent(self, (x, y))
        #Healthworkers can move always
        else:
            x = self.random.randrange(self.model.grid.width)
            y = self.random.randrange(self.model.grid.height)
            self.model.grid.move_agent(self, (x, y))

    def step(self):
        if self.status is not AgentStatus.dead:
            self.move()
            self.update_infected()
            self.update_recovered()
            self.update_susceptible()
            self.set_household(self.household)
            self.random_activation()
            self.check_for_health_worker()
            self.update_dead()

class Household:
    def __init__(self, unique_id, size):
        self.unique_id = unique_id
        self.size = size
        self.individuals = []

    def add_individual(self, agent):
        self.individuals.append(agent)

    def __len__(self):
        return len(self.individuals)

class CovidModel(Model):

    def __init__(self, no_agents, width, height,
                 init_infected, infection_period,
                 immunity_period, mortality_rate,
                 lockdown_status, protective_measures,
                 perc_health_worker, household_size,
                 isolation_capacity):
        self.no_agents = no_agents
        self.grid = MultiGrid(width, height, False)
        self.init_infected = init_infected
        if protective_measures == ProtectiveMeasures.no_measure.value:
            self.infection_rate = 0.07
        elif protective_measures == ProtectiveMeasures.masks_mandatory.value:
            self.infection_rate = 0.035
        elif protective_measures == ProtectiveMeasures.social_distancing.value:
            self.infection_rate = 0.05
        elif protective_measures == ProtectiveMeasures.both.value:
            self.infection_rate = 0.01
        self.infection_period = infection_period
        self.immunity_period = immunity_period
        self.mortality_rate = mortality_rate
        self.lockdown_status = lockdown_status
        self.protective_measures = protective_measures
        self.perc_health_worker = perc_health_worker
        self.household_size = household_size
        self.household_infection_rate_factor = 0.07
        self.isolation_capacity = isolation_capacity
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.no_agents):
            temperature = random.uniform(97.0, 99.0)
            blood_oxygen_level = random.randint(90, 100)
            a = Agent(i, self, self.lockdown_status, self.infection_rate,temperature,blood_oxygen_level)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Set up households
        self.setup_households()

        # Collect count of susceptible, infected, and recovered agents
        self.datacollector = DataCollector({
            'Susceptible': 'susceptible',
            'Infected': 'infected',
            'Recovered & Immune': 'immune',
            'Isolated': 'isolated',
            'Dead': 'dead',
            '1_Member_House': 'household_size_distribution_1',
            '2_Member_House': 'household_size_distribution_2',
            '3_Member_House': 'household_size_distribution_3',
            '4_Member_House': 'household_size_distribution_4',
            '5_Member_House': 'household_size_distribution_5',
            '6_Member_House': 'household_size_distribution_6',
        })

        self.restrictions = pd.DataFrame(pd.read_csv(r'Restrictions.csv'))
        self.restrictions['Date'] = pd.to_datetime(self.restrictions['Date'], format='%d-%m-%Y')
        self.restrictions['NextDate'] = pd.to_datetime(self.restrictions['NextDate'], format='%d-%m-%Y')
        self.restrictions['Steps'] = (self.restrictions['NextDate'] - self.restrictions['Date']).dt.days
        self.restrictions['running_sum'] = self.restrictions['Steps'].cumsum()
        self.restrictions['running_sum'] = self.restrictions.running_sum.shift(1)
        self.restrictions['running_sum'] = self.restrictions['running_sum'].fillna(1)

    def setup_households(self):
        # Create households and assign agents to them
        self.households = []
        # Set the distribution percentages
        dist = [0.356, 0.339, 0.167, 0.092, 0.036, 0.011]
        # Set the total number of households
        num_households = self.no_agents * 0.65
        # Calculate the number of households for each size category
        sizes = [int(round(p * num_households)) for p in dist]
        # Adjust the number of households to match the total
        sizes[0] += int(round(num_households - sum(sizes)))
        # Generate a list of household sizes
        household_list = []
        for i, size in enumerate(sizes):
            household_list.extend([i + 1] * size)
        # Shuffle the list to randomize the order
        random.shuffle(household_list)

        for index, size in enumerate(household_list):
            self.households.append(Household(index, size))

        print(f"sizes are {sizes} and total is {sum(sizes)}")
        houses = 0
        agent_num = 0
        agent_array = self.schedule.agents
        for house in self.households:
            house.individuals = agent_array[:house.size]
            for agent in agent_array[:house.size]:
                agent.set_household(house)
                agent_num += 1
            agent_array = agent_array[house.size:]
            houses += 1
        print(f"Total number of houses is {houses}")
        print(f"Total placed agents are {agent_num}")

    @property
    def susceptible(self):
        agents = self.schedule.agents
        susceptible = [a for a in agents if a.status == AgentStatus.susceptible]
        return int(len(susceptible))

    @property
    def infected(self):
        agents = self.schedule.agents
        infected = [a for a in agents if a.status == AgentStatus.infected]
        return int(len(infected))

    @property
    def immune(self):
        agents = self.schedule.agents
        immune = [a for a in agents if a.status == AgentStatus.recovered]
        return int(len(immune))

    @property
    def isolated(self):
        agents = self.schedule.agents
        isolated = [a for a in agents if a.status == AgentStatus.isolated]
        return int(len(isolated))

    @property
    def dead(self):
        agents = self.schedule.agents
        dead = [a for a in agents if a.status == AgentStatus.dead]
        return int(len(dead))

    @property
    def household_size_distribution_1(self):
        house_num = [household for household in self.households if household.size == 1]
        return len(house_num)

    @property
    def household_size_distribution_2(self):
        house_num = [household for household in self.households if household.size == 2]
        return len(house_num)

    @property
    def household_size_distribution_3(self):
        house_num = [household for household in self.households if household.size == 3]
        return len(house_num)

    @property
    def household_size_distribution_4(self):
        house_num = [household for household in self.households if household.size == 4]
        return len(house_num)

    @property
    def household_size_distribution_5(self):
        house_num = [household for household in self.households if household.size == 5]
        return len(house_num)

    @property
    def household_size_distribution_6(self):
        house_num = [household for household in self.households if household.size == 6]
        return len(house_num)

    def step(self):
        if self.schedule.steps >= max(self.restrictions['running_sum']):
            self.running = False
        else:
            result_df = self.restrictions[self.restrictions['running_sum'] == self.schedule.steps]
            if result_df.empty == False:
                print(result_df['lockdown_status'].values[0], result_df['protective_measures'].values[0])
                self.lockdown_status = result_df['lockdown_status'].values[0]
                self.protective_measures = result_df['protective_measures'].values[0]
        if self.protective_measures == ProtectiveMeasures.no_measure.value:
            self.infection_rate = 0.07
        elif self.protective_measures == ProtectiveMeasures.masks_mandatory.value:
            self.infection_rate = 0.035
        elif self.protective_measures == ProtectiveMeasures.social_distancing.value:
            self.infection_rate = 0.05
        elif self.protective_measures == ProtectiveMeasures.both.value:
            self.infection_rate = 0.01
        #randomly infect if infection is too low
        active_agents = self.schedule.agents
        alive_agents = [a for a in active_agents if a.status is not AgentStatus.dead]
        infected = [a for a in active_agents if a.status == AgentStatus.infected]
        if (int(len(infected))/int(len(alive_agents)) < self.init_infected *100):
            for a in alive_agents:
                a.random_activation()


        self.datacollector.collect(self)
        self.schedule.step()
        self.save_results()
        

    def save_results(self):
        # Save results to CSV file
        df = self.datacollector.get_model_vars_dataframe()
        df.to_csv('simulation_results.csv', index=True)
