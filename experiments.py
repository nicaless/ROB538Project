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
B_skill = 10
B_num = 3

ROWS = 10
COLS = 10

MAX_TIME_STEPS = 50

FAILURE_RATE = int((MAX_TIME_STEPS / N) * 2)
# print(FAILURE_RATE)

# Initial Configurations to Test
C1 = {'defuser': 2, 'search': 5, 'detection': 3}
C2 = {'defuser': 3, 'search': 3, 'detection': 4}
C3 = {'defuser': 3, 'search': 2, 'detection': 5}
C4 = {'defuser': 5, 'search': 2, 'detection': 3}
init_configs = 4
configurations = [C1, C2, C3, C4]
configuration_results = {i: {'num_failures': [], 'bombs_defused': [], 'timesteps': []} for i in range(1, 5)}

max_configurations = 6
TRIALS = 10
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
        b1 = Bomb(np.array([0, 9]), B_skill)
        b2 = Bomb(np.array([9, 9]), B_skill)
        b3 = Bomb(np.array([9, 0]), B_skill)
        bombs = [b1, b2, b3]


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
            grid.plot_state(i)
            if ((i+1) % FAILURE_RATE) == 0:
                # print('Agent Failure')
                count_failures += 1
                a = np.random.choice(N)
                grid.agents[a].failed = True
                config[grid.agents[a].type_name] -= 1
                for a in grid.agents:
                    a.get_team_config(config)


        # print(i)
        # grid.print_state()
        # grid.plot_state(i)
        total_steps = i
        # print('Total Failures {}'.format(count_failures))

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

    # TODO: create new configurations based on overall trial feedback
    # new_c = add_new_config(agent_feedback, dict(C))
    # if new_c is not None:
    #     print('Adding New Configuration')
    #     configurations.append(new_c)
    #     configuration_results[init_configs+1] = {'num_failures': 0, 'bombs_defused': 0, 'timesteps': 0}
    #     init_configs += 1


##############################################################################
# Save Results
##############################################################################
print(configuration_results)
print(best_success_rate)
print(best_config)

# TODO: run best found config again to generate gif
# with imageio.get_writer('mygif.gif', mode='I') as writer:
#     for i in range(0, total_steps):
#         filename = 'plots/{}.png'.format(i)
#         image = imageio.imread(filename)
#         writer.append_data(image)