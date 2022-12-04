import numpy as np
import imageio

from agent import Agent
from grid_world import Bomb, GridWorld
from matplotlib import pyplot as plt

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
agent_defusal_types = {'defuser':  5, 'search': 3, 'detection': 4}

N = 20   # maximum team number
B_skill = 10
B_num = 5

ROWS = 10
COLS = 10

MAX_TIME_STEPS = 60
de = 1
se = 1
det =1
plde = []
plse = []
pldet = []
bdefused = []
failures = []
nosteps = []
N = de + se + det
FAILURE_RATE = 10
print(FAILURE_RATE)
for zz in range(10):
    N = de + se + det
    #FAILURE_RATE = int((MAX_TIME_STEPS / N) * 2)
    # Initial Configurations to Test
    plde.append(de)
    plse.append(se)
    pldet.append(det)
    C = {'defuser': de, 'search': se, 'detection': det}
    de += 1
    se += 1
    det += 1

    # Build Team Based on Configuration:
    agents = []
    for type_name, kn in C.items():
        agent_template = agent_types[type_name]
        for k in range(kn):
            a = Agent(agent_template['init_pos'], type_name, agent_defusal_types,
                      agent_template['defusal_skill'], agent_template['mobility'],
                      agent_template['sensing'], agent_template['eps'])
            a.get_team_config(dict(C))
            agents.append(a)


    # Initialize Bombs at Opposite Corners
    b1 = Bomb(np.array([0, 9]), B_skill)
    b2 = Bomb(np.array([9, 9]), B_skill)
    b3 = Bomb(np.array([9, 0]), B_skill)
    b4 = Bomb(np.array([5, 5]), B_skill)
    b5 = Bomb(np.array([0, 5]), B_skill)
    bombs = [b1, b2, b3, b4, b5]


    grid = GridWorld(agents, bombs)
    B_num = len(bombs)
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

    print(i, " ##### ")
    nosteps.append(i)
    bdefused.append(grid.global_reward)
    #grid.print_state()
    #grid.plot_state(i)
    total_steps = i
    print('Total Failures {}'.format(count_failures))
    failures.append(count_failures)

plt.figure(figsize=(9, 9))
plt.plot(plde, nosteps, marker='o', label = 'no. of steps')
plt.plot(plde, bdefused, marker='v', label = 'bombs defused')
plt.plot(plde, failures, marker='p', label = 'total failures')
plt.xticks(np.arange(0, np.max(plde)+1, 1))
plt.yticks(np.arange(0, np.max(nosteps)+2, 5))
plt.xlabel('Number of agents of each type')
plt.ylabel('Number of steps')
#plt.savefig('./randid.png')
plt.legend()
plt.show()