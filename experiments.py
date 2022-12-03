import numpy as np
import imageio

from agent import Agent
from grid_world import Bomb, GridWorld


INIT_POS = np.array([0, 0])
EPS = 0.5


agent_types = {'defuser': {'init_pos': INIT_POS,
                           'defusal_skill': 5,
                           'mobility': 1,
                           'sensing': 1,
                           'eps': EPS},
               'search': {'init_pos': INIT_POS,
                           'defusal_skill': 3,
                           'mobility': 5,
                           'sensing': 3,
                           'eps': EPS},
               'detection': {'init_pos': INIT_POS,
                           'defusal_skill': 4,
                           'mobility': 3,
                           'sensing': 5,
                           'eps': EPS}}

N = 10   # maximum team number
B_skill = 10
B_num = 3

ROWS = 10
COLS = 10

MAX_TIME_STEPS = 50

FAILURE_RATE = int((MAX_TIME_STEPS / N) * 2)
print(FAILURE_RATE)

# Initial Configurations to Test
C = {'defuser': 2, 'search': 5, 'detection': 3}


# Build Team Based on Configuration:
agents = []
for type_name, kn in C.items():
    agent_template = agent_types[type_name]
    for k in range(kn):
        a = Agent(agent_template['init_pos'], agent_template['defusal_skill'],
                  agent_template['mobility'], agent_template['sensing'],
                  agent_template['eps'])
        agents.append(a)


# Initialize Bombs at Opposite Corners
b1 = Bomb(np.array([0, 9]), B_skill)
b2 = Bomb(np.array([9, 9]), B_skill)
b3 = Bomb(np.array([9, 0]), B_skill)
bombs = [b1, b2, b3]


grid = GridWorld(agents, bombs)

count_failures = 0
for i in range(MAX_TIME_STEPS):
    grid.step()
    if grid.global_reward >= B_num:
        break
    grid.plot_state(i)
    if ((i+1) % FAILURE_RATE) == 0:
        print('Agent Failure')
        count_failures += 1
        a = np.random.choice(N)
        grid.agents[a].failed = True

print(i)
grid.print_state()
grid.plot_state(i)
total_steps = i
print('Total Failures {}'.format(count_failures))

# with imageio.get_writer('mygif.gif', mode='I') as writer:
#     for i in range(0, total_steps):
#         filename = 'plots/{}.png'.format(i)
#         image = imageio.imread(filename)
#         writer.append_data(image)