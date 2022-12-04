import numpy as np
import imageio

from agent import Agent
from grid_world import Bomb, GridWorld, add_new_config


##############################################################################
# Experiment Parameters
##############################################################################

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


N = 10   # maximum team number
B_skill = 20
bomb_positions = [np.array([0, 9]), np.array([9, 9]), np.array([9, 0])]
# bomb_positions = [np.array([0, 9]), np.array([9, 9]), np.array([9, 0]), np.array([5, 5]), np.array([9, 5])]
# bomb_positions = [np.array([0, 9]), np.array([9, 9]), np.array([9, 0]), np.array([5, 5]), np.array([9, 5]),
#                   np.array([0, 5]), np.array([5, 9]), np.array([7, 7]), np.array([0, 7]), np.array([7, 0])]
B_num = len(bomb_positions)

ROWS = 10
COLS = 10

MAX_TIME_STEPS = 50

# FAILURE_RATE = int((MAX_TIME_STEPS / N) * 2)
FAILURE_RATE = 2
# print(FAILURE_RATE)

# Initial Configurations to Test
# N = 10
# C1 = {'defuser': 2, 'search': 5, 'detection': 3}
# C2 = {'defuser': 2, 'search': 3, 'detection': 5}
# C3 = {'defuser': 3, 'search': 2, 'detection': 5}
# C4 = {'defuser': 3, 'search': 5, 'detection': 2}
#
# C5 = {'defuser': 5, 'search': 2, 'detection': 3}
# C6 = {'defuser': 5, 'search': 3, 'detection': 2}
# C7 = {'defuser': 4, 'search': 3, 'detection': 3}
# C8 = {'defuser': 3, 'search': 4, 'detection': 3}
# C9 = {'defuser': 3, 'search': 3, 'detection': 4}
# C10 = {'defuser': 6, 'search': 2, 'detection': 2}

# N = 30
# C1 = {'defuser': 10, 'search': 10, 'detection': 10}
# C2 = {'defuser': 15, 'search': 10, 'detection': 5}
# C3 = {'defuser': 15, 'search': 5, 'detection': 10}
# C4 = {'defuser': 10, 'search': 5, 'detection': 15}
#
# C5 = {'defuser': 10, 'search': 15, 'detection': 5}
# C6 = {'defuser': 5, 'search': 10, 'detection': 15}
# C7 = {'defuser': 5, 'search': 15, 'detection': 10}
# C8 = {'defuser': 12, 'search': 8, 'detection': 10}
# C9 = {'defuser': 12, 'search': 10, 'detection': 8}
# C10 = {'defuser': 10, 'search': 8, 'detection': 12}

N = 40
C1 = {'defuser': 30, 'search': 5, 'detection': 5}
C2 = {'defuser': 15, 'search': 15, 'detection': 10}
C3 = {'defuser': 15, 'search': 10, 'detection': 15}

# init_configs = 10
# configurations = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10]
# saved_configurations = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10]
# configuration_results = {i: {'num_failures': [], 'bombs_defused': [], 'timesteps': []} for i in range(1, 11)}

init_configs = 3
configurations = [C1, C2, C3]
saved_configurations = [C1, C2, C3]
configuration_results = {i: {'num_failures': [], 'bombs_defused': [], 'timesteps': []} for i in range(1, 4)}

max_configurations = 10
TRIALS = 20
OPT = False

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
                a = Agent(agent_template['init_pos'], type_name, agent_defusal_types,
                          agent_template['defusal_skill'], agent_template['mobility'],
                          agent_template['sensing'], agent_template['eps'])
                a.get_team_config(dict(C))
                agents.append(a)


        # Initialize Bombs at Opposite Corners
        bombs = []
        for pos in bomb_positions:
            bombs.append(Bomb(pos, B_skill))

        grid = GridWorld(agents, bombs)
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
                a = np.random.choice(N)
                grid.agents[a].failed = True
                config[grid.agents[a].type_name] -= 1
                for a in grid.agents:
                    a.get_team_config(config)

        total_steps = i

        configuration_results[configurations_tried]['num_failures'].append(count_failures)
        configuration_results[configurations_tried]['bombs_defused'].append(grid.global_reward)
        if grid.global_reward == B_num:
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
        new_c = add_new_config(agent_feedback, dict(C))
        if (new_c is not None) and OPT:
            print('Adding New Configuration')
            configurations.append(new_c)
            saved_configurations.append(new_c)
            configuration_results[init_configs+1] = {'num_failures': [], 'bombs_defused': [], 'timesteps': []}
            init_configs += 1


##############################################################################
# Save Results
##############################################################################
RESULTS_FILE = open('N{}_B{}_F{}{}.txt'.format(N, B_num, FAILURE_RATE, '_OPT' if OPT else ''), 'w')

RESULTS_FILE.write('###### Experiment Results \n')
RESULTS_FILE.write('###### Total Configurations Tested {}\n'.format(configurations_tried))

RESULTS_FILE.write('##### Configuration Results\n')
for c, res in configuration_results.items():
    RESULTS_FILE.write('Configuration {}\n'.format(c))
    RESULTS_FILE.write('Defuser Agents {}\n'.format(saved_configurations[c-1]['defuser']))
    RESULTS_FILE.write('Search Agents {}\n'.format(saved_configurations[c-1]['search']))
    RESULTS_FILE.write('Detection Agents {}\n'.format(saved_configurations[c-1]['detection']))
    RESULTS_FILE.write('Results {}\n'.format(c))
    RESULTS_FILE.write('Avg Num Failures {}\n'.format(np.mean(res['num_failures'])))
    RESULTS_FILE.write('Avg Bombs Defused {}\n'.format(np.mean(res['bombs_defused'])))
    RESULTS_FILE.write('Avg Timesteps to Defuse All Bombs {}\n\n'.format(np.mean(res['timesteps'])))

RESULTS_FILE.write('###### Best Configuration\n')
RESULTS_FILE.write('Defuser Agents {}\n'.format(best_config['defuser']))
RESULTS_FILE.write('Search Agents {}\n'.format(best_config['search']))
RESULTS_FILE.write('Detection Agents {}\n'.format(best_config['detection']))
RESULTS_FILE.write('Success Rate: {}\n'.format(best_success_rate))
RESULTS_FILE.close()

# TODO: run best found config again to generate gif
# with imageio.get_writer('mygif.gif', mode='I') as writer:
#     for i in range(0, total_steps):
#         filename = 'plots/{}.png'.format(i)
#         image = imageio.imread(filename)
#         writer.append_data(image)