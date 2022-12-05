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
B_num = 3

ROWS = 50
COLS = 50

MAX_TIME_STEPS = 500
de = 1
se = 1
det = 1
plde = []
plse = []
pldet = []
bdefused = []
failures = []
nosteps = []
N = de + se + det
FAILURE_RATE = 50
print(FAILURE_RATE)
for zz in range(30):
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
            a = Agent(np.random.choice(50, size=(1, 2))[0], type_name, agent_defusal_types,
                      agent_template['defusal_skill'], agent_template['mobility'],
                      agent_template['sensing'], agent_template['eps'])
            a.get_team_config(dict(C))
            agents.append(a)


    # Initalize Bombs Randomly
    bomb_locs = np.random.choice(50, size=(B_num, 2))
    bombs = []
    for loc in bomb_locs:
        bombs.append(Bomb(loc, B_skill))


    grid = GridWorld(agents, bombs)
    B_num = len(bombs)
    count_failures = 0
    for i in range(MAX_TIME_STEPS):
        grid.step()
        if grid.global_reward >= B_num:
            break
        # grid.plot_state(i)
        if ((i+1) % FAILURE_RATE) == 0:
            # print('Agent Failure')
            count_failures += 1
            a = np.random.choice(N)
            grid.agents[a].failed = True

    # print(i, " ##### ")
    nosteps.append(i)
    bdefused.append(grid.global_reward)
    #grid.print_state()
    # grid.plot_state(i)
    total_steps = i
    # print('Total Failures {}'.format(count_failures))
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