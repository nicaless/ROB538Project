import numpy as np
import imageio

from agent import Agent
from grid_world import Bomb, GridWorld, add_new_config



##############################################################################
# Experiment Parameters
##############################################################################
np.random.seed(42)
INIT_POS = np.array([0, 0])
EPS = 0.2


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


B_skill = 15
B_num = 5

ROWS = 100
COLS = 100

FAILURE_RATE = 60

# Initial Configurations to Test
# N = 9
# C1 = {'defuser': 7, 'search': 1, 'detection': 1}
# C2 = {'defuser': 1, 'search': 1, 'detection': 7}
# C3 = {'defuser': 1, 'search': 7, 'detection': 1}
# C4 = {'defuser': 3, 'search': 3, 'detection': 3}



# N = 10
# C1 = {'defuser': 8, 'search': 1, 'detection': 1}
# C2 = {'defuser': 7, 'search': 1, 'detection': 2}
# C3 = {'defuser': 7, 'search': 2, 'detection': 1}
# C1 = {'defuser': 4, 'search': 3, 'detection': 3}
# C2 = {'defuser': 3, 'search': 3, 'detection': 4}
# C3 = {'defuser': 3, 'search': 4, 'detection': 3}


N = 30
C1 = {'defuser': 28, 'search': 1, 'detection': 1}
C2 = {'defuser': 1, 'search': 28, 'detection': 1}
C3 = {'defuser': 1, 'search': 1, 'detection': 28}
C4 = {'defuser': 10, 'search': 10, 'detection': 10}
# C1 = {'defuser': 8, 'search': 10, 'detection': 12}
# C2 = {'defuser': 12, 'search': 8, 'detection': 10}
# C3 = {'defuser': 10, 'search': 12, 'detection': 8}


# N = 40
# C1 = {'defuser': 15, 'search': 10, 'detection': 15}
# C3 = {'defuser': 5, 'search': 30, 'detection': 5}
# C2 = {'defuser': 18, 'search': 10, 'detection': 12}
#
# C4 = {'defuser': 12, 'search': 15, 'detection': 13}
# C5 = {'defuser': 15, 'search': 13, 'detection': 12}
# C6 = {'defuser': 13, 'search': 12, 'detection': 15}

# N = 51
# C1 = {'defuser': 47, 'search': 2, 'detection': 2}
# C2 = {'defuser': 2, 'search': 2, 'detection': 47}
# C3 = {'defuser': 2, 'search': 47, 'detection': 2}
# C4 = {'defuser': 17, 'search': 17, 'detection': 17}


# init_configs = 10
# configurations = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10]
# saved_configurations = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10]
# configuration_results = {i: {'num_failures': [], 'bombs_defused': [], 'timesteps': []} for i in range(1, 11)}

MAX_TIME_STEPS = N*FAILURE_RATE

init_configs = 4
configurations = [C1, C2, C3, C4]
saved_configurations = [C1, C2, C3, C4]
configuration_results = {i: {'num_failures': [], 'bombs_defused': [], 'timesteps': []} for i in range(1, init_configs+1)}

max_configurations = 10
TRIALS = 10
OPT = True

RESULTS_FILE = open('results/N{}_B{}_F{}{}.txt'.format(N, B_num, FAILURE_RATE, '_OPT' if OPT else ''), 'w')

##############################################################################
# Experiment Setup
##############################################################################
configurations_tried = 0
best_config = C1
best_success_rate = 0

while (len(configurations) > 0) and (configurations_tried <= max_configurations):
    print('Configurations Tried {}'.format(configurations_tried))
    configurations_tried += 1
    C = configurations.pop(0)
    trial_feedback = []
    for t in range(TRIALS):
        # Build Team Based on Configuration:
        agents = []
        for type_name, kn in C.items():
            agent_template = agent_types[type_name]
            for k in range(kn):
                a = Agent(np.random.choice(ROWS-1, size=(1, 2))[0], type_name, agent_defusal_types,
                          agent_template['defusal_skill'], agent_template['mobility'],
                          agent_template['sensing'], agent_template['eps'], bounds=(ROWS, COLS))
                a.get_team_config(dict(C))
                agents.append(a)

        # Initalize Bombs Randomly
        bomb_locs = np.random.choice(ROWS-1, size=(B_num, 2))
        bombs = []
        for loc in bomb_locs:
            bombs.append(Bomb(loc, B_skill))

        grid = GridWorld(agents, bombs, bounds=(ROWS, COLS))
        config = dict(C)

##############################################################################
# Run Experiments
##############################################################################
        count_failures = 0
        for i in range(MAX_TIME_STEPS):
            grid.step()
            if grid.global_reward >= B_num:
                break
            # grid.plot_state(i)
            if ((i+1) % FAILURE_RATE) == 0:
                # print('Agent Failure')
                count_failures += 1
                a = np.random.choice(N-1)
                grid.agents[a].failed = True
                config[grid.agents[a].type_name] -= 1
                for a in grid.agents:
                    a.get_team_config(config)

        total_steps = i

        configuration_results[configurations_tried]['num_failures'].append(count_failures)
        configuration_results[configurations_tried]['bombs_defused'].append(grid.global_reward)
        configuration_results[configurations_tried]['timesteps'].append(total_steps)

        agent_feedback = []
        for a in agents:
            agent_feedback.append(a.add_types)

        trial_feedback = trial_feedback + agent_feedback

    success_rate = np.mean(configuration_results[configurations_tried]['bombs_defused'])

    if success_rate >= best_success_rate:
        best_success_rate = success_rate
        best_config = C

        # Create new configurations based on overall trial feedback
        new_c = add_new_config(agent_feedback, dict(C), max_to_add=5, max_team=N)
        if (new_c is not None) and OPT:
            if new_c not in saved_configurations:
                print('Adding New Configuration')
                configurations.append(new_c)
                saved_configurations.append(new_c)
                configuration_results[init_configs+1] = {'num_failures': [], 'bombs_defused': [], 'timesteps': []}
                init_configs += 1


##############################################################################
# Save Results
##############################################################################
RESULTS_FILE = open('results/N{}_B{}_F{}{}.txt'.format(N, B_num, FAILURE_RATE, '_OPT' if OPT else ''), 'w')

RESULTS_FILE.write('###### Experiment Results \n')
RESULTS_FILE.write('###### Total Configurations Tested {}\n'.format(configurations_tried))

RESULTS_FILE.write('##### Configuration Results\n')

means_ntime = []
means_failures = []
means_bdfuse = []
list_de = []
list_se = []
list_det = []

labels = []
i = 0
for c, res in configuration_results.items():
    i += 1
    RESULTS_FILE.write('Configuration {}\n'.format(c))
    RESULTS_FILE.write('Defuser Agents {}\n'.format(saved_configurations[c-1]['defuser']))
    RESULTS_FILE.write('Search Agents {}\n'.format(saved_configurations[c-1]['search']))
    RESULTS_FILE.write('Detection Agents {}\n'.format(saved_configurations[c-1]['detection']))
    RESULTS_FILE.write('Results {}\n'.format(c))
    RESULTS_FILE.write('Avg Num Failures {}\n'.format(np.mean(res['num_failures'])))
    RESULTS_FILE.write('Avg Bombs Defused {}\n'.format(np.mean(res['bombs_defused'])))
    RESULTS_FILE.write('Avg Timesteps to Defuse All Bombs {}\n\n'.format(np.mean(res['timesteps'])))
    means_ntime.append(np.mean(res['timesteps']))
    means_failures.append(np.mean(res['num_failures']))
    means_bdfuse.append(np.mean(res['bombs_defused']))
    list_de.append(saved_configurations[c-1]['defuser'])
    list_se.append(saved_configurations[c-1]['search'])
    list_det.append(saved_configurations[c-1]['detection'])
    labels.append(str(i))

RESULTS_FILE.write('###### Best Configuration\n')
RESULTS_FILE.write('Defuser Agents {}\n'.format(best_config['defuser']))
RESULTS_FILE.write('Search Agents {}\n'.format(best_config['search']))
RESULTS_FILE.write('Detection Agents {}\n'.format(best_config['detection']))
RESULTS_FILE.write('Success Rate: {}\n'.format(best_success_rate))
RESULTS_FILE.close()

import matplotlib.pyplot as plt

eans_ntime = np.array(means_ntime)
eans_failures = np.array(means_failures)
eans_bdfuse = np.array(means_bdfuse)
ist_de = np.array(list_de)
ist_se = np.array(list_se)
ist_det = np.array(list_det)

for i in range(len(means_ntime)):
    ist_de[i] *= (means_ntime[i] / N)
    ist_se[i] *= (means_ntime[i] / N)
    ist_det[i] *= (means_ntime[i] / N)

width = 0.35       # the width of the bars: can also be len(x) sequence



# Unmesh's Plot - Bar Plots where height is the metric, splits are the team makeup, x-axis is different configurations
# fig, ax = plt.subplots()
#
# ax.bar(labels, ist_se, width, label='search')
# ax.bar(labels, ist_det, width, bottom=ist_se, label='detection')
#
# ax.bar(labels, ist_de, width, bottom=ist_se+ist_det, label='defuser')
# ax.set_ylabel('Avg Number of Timesteps to Defuse All Bombs')
#
# plt.savefig('team_size_{}_avg_time.png')
# plt.clf()
#
# for i in range(len(means_bdfuse)):
#     ist_de[i] *= (means_bdfuse[i] / N)
#     ist_se[i] *= (means_bdfuse[i] / N)
#     ist_det[i] *= (means_bdfuse[i] / N)
#
# fig, ax = plt.subplots()
# ax.bar(labels, ist_se, width, label='search')
# ax.bar(labels, ist_det, width, bottom=ist_se, label='detection')
#
# ax.bar(labels, ist_de, width, bottom=ist_se+ist_det, label='defuser')
# ax.set_ylabel('Avg Number of Bombs Defused')
#
# plt.savefig('team_size_{}_avg_bombs.png')


fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)

ax3.bar(labels, list_se, width, label='search', alpha=0.5)
ax3.bar(labels, list_det, width, bottom=list_se, label='detection', alpha=0.5)
ax3.bar(labels, list_de, width, bottom=list(np.array(list_se) + np.array(list_det)), label='defuser', alpha=0.5)

ax3.set_ylabel('# Agents')
ax3.set_xlabel('Configuration #')

ax1.plot(labels, means_ntime)
ax1.set_ylabel('Avg Time to Defuse All Bombs')

ax2.plot(labels, means_bdfuse)
ax2.set_ylabel('Avg # Defused')

# ax1.title('Team Size {}'.format(N))

ax3.legend()
plt.savefig('poi_num_{}.png'.format(B_num))

plt.clf()