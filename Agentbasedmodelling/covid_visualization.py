from mesa.visualization.modules import CanvasGrid, ChartModule, BarChartModule
from mesa.visualization.ModularVisualization import ModularServer
from covid_model import *
from covid_model import Household
from covid_model import CovidModel, Household

def agent_portrayal(agent):
    portrayal = {
        'Shape': 'circle',
        'Layer': 0,
        'r': 1,
        'Color': 'lightblue'}

    if agent.status == AgentStatus.infected:
        portrayal['Color'] = 'red'
        portrayal['Filled'] = True
        portrayal["Layer"] = 1
        portrayal["scale"] = 0.9

    if agent.status == AgentStatus.isolated:
        portrayal['Color'] = 'grey'
        portrayal['Filled'] = True
        portrayal["Layer"] = 1
        portrayal["scale"] = 0.9

    if agent.status == AgentStatus.recovered:
        portrayal['Color'] = 'green'
        portrayal["Layer"] = 2
        portrayal["scale"] = 0.9

    if agent.health_worker == True:
        portrayal['Color'] = 'yellow'
        portrayal["Layer"] = 3
        portrayal["scale"] = 0.9

    if agent.status == AgentStatus.dead:
        portrayal['Color'] = 'black'
        portrayal["Layer"] = 0
        portrayal["scale"] = 0.9

    return portrayal

grid = CanvasGrid(agent_portrayal, 141, 141, 1410, 1410)

line_charts = ChartModule([
    {'Label': 'Susceptible', 'Color': 'lightblue'},
    {'Label': 'Infected', 'Color': 'red'},
    {'Label': 'Recovered & Immune', 'Color': 'green'},
    {'Label': 'Isolated', 'Color': 'grey'},
    {'Label': 'Dead', 'Color': 'black'}])

bar_chart = BarChartModule([{'Label': '1_Member_House', 'Color': 'red'},
                            {'Label': '2_Member_House', 'Color': 'green'},
                            {'Label': '3_Member_House', 'Color': 'blue'},
                            {'Label': '4_Member_House', 'Color': 'lightblue'},
                            {'Label': '5_Member_House', 'Color': 'grey'},
                            {'Label': '6_Member_House', 'Color': 'black'}])

server = ModularServer(CovidModel,
                       [grid, line_charts, bar_chart],
                       'COVID Simulation Model',
                       model_params)

server.port = 8521  # default port if unspecified
server.launch()
