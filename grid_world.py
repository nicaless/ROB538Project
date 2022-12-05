from matplotlib import pyplot as plt, patches
import numpy as np
import operator

from agent import Agent

ROWS = 50
COLS = 50


def add_new_config(agent_feedback, current_config, max_to_add=5, max_team=30):
    # find agent type that has the most votes (highest value in agent's add_type dictionary)
    # get average number of additions for that agent type
    # uniformly subtract that number from the other agent types
    votes = {atype: 0 for atype in current_config.keys()}
    additions = {atype: [] for atype in current_config.keys()}
    for feedback in agent_feedback:
        highest_vote = max(feedback.items(), key=operator.itemgetter(1))[0]
        if feedback[highest_vote] == 0:
            continue
        votes[highest_vote] += 1
        additions[highest_vote].append(feedback[highest_vote])

    winner = max(votes.items(), key=operator.itemgetter(1))[0]
    if len(additions[winner]) == 0:
        return None
    total_to_add = int(max(0, min(min(max_team - current_config[winner], max_to_add),
                                  np.mean(additions[winner]))))

    # print(winner)
    # print('adding total')
    # print(total_to_add)
    #
    # print('current_config')
    # print(current_config)
    new_config = dict(current_config)
    new_config[winner] += total_to_add
    while total_to_add > 0:
        atype = np.random.choice(list(current_config.keys()))
        if atype == winner:
            continue
        if new_config[atype] == 0:
            continue
        new_config[atype] -= 1
        total_to_add -= 1
        # print(total_to_add)
        # print(new_config)

    return new_config



class Bomb:
    def __init__(self, position, skill_level):
        self.position = position
        self.skill_level = skill_level
        self.defused = False


class GridWorld:
    def __init__(self, agents, bombs, bounds=(ROWS, COLS)):
        self.agents = agents
        self.bombs = bombs
        self.grid_state =  np.zeros((bounds[0], bounds[1]))
        self.ROWS = bounds[0]
        self.COLS = bounds[1]
        self.update_state()
        self.global_reward = 0


    def print_state(self):
        print(self.grid_state)
        print('Bombs Defused: {}'.format(self.global_reward))

    def plot_state(self, time_step):
        # Plot Agents
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlim(-2, self.ROWS+2)
        ax.set_ylim(-2, self.COLS+2)
        for agent in self.agents:
            if not agent.failed:
                pos = agent.position
                circle1 = patches.Circle((pos[0], pos[1]), radius=agent.sensing, color='green', alpha=0.1)
                ax.scatter(pos[0], pos[1], color='blue', alpha=0.1)
                ax.add_patch(circle1)

        # Plot Bombs
            for bomb in self.bombs:
                if not bomb.defused:
                    pos = bomb.position
                    ax.scatter(pos[0], pos[1], color='red', alpha=0.5)

        plt.savefig('plots/{}.png'.format(time_step))
        plt.clf()
        plt.close()


    def step(self):
        # Move Agents
        for i, agent in enumerate(self.agents):
            # print('Moving Agent {}'.format(i))
            agent.step(self.grid_state)

        self.update_state()

        # Defuse Any Bombs
        agents_available_to_defuse = []
        bomb_states = {}
        bomb_skill_level = {}
        for bomb in self.bombs:
            if bomb.defused:
                continue
            agents_at_bomb = []
            for agent in self.agents:
                if agent.failed:
                    continue
                if np.array_equal(agent.position, bomb.position):
                    agents_at_bomb.append(agent)
                    agents_available_to_defuse.append(agent)
            total_agent_skill = np.sum([agent.defusal_skill for agent in agents_at_bomb])
            if total_agent_skill >= bomb.skill_level:
                bomb.defused = True
                self.global_reward += 1
            bomb_states[(bomb.position[0], bomb.position[1])] = agents_at_bomb
            bomb_skill_level[(bomb.position[0], bomb.position[1])] = bomb.skill_level

        # agents_available_to_defuse = set(agents_available_to_defuse)
        for i, agent in enumerate(self.agents):
            # print('Updating Values Agent {}'.format(i))
            agent.update_probabilities(len(self.agents), self.global_reward,
                                       bomb_states, bomb_skill_level)

        self.update_state()


    def update_state(self):
        grid = np.zeros((self.ROWS, self.COLS))

        for agent in self.agents:
            if not agent.failed:
                pos = agent.position
                grid[pos[0], pos[1]] = 1

        for bomb in self.bombs:
            if not bomb.defused:
                pos = bomb.position
                grid[pos[0], pos[1]] = 2

        self.grid_state = grid


if __name__ == "__main__":
    a1 = Agent(np.array([1, 1]), 3, 1, 5, 0.9)
    a2 = Agent(np.array([0, 0]), 3, 1, 5, 0.9)
    b1 = Bomb(np.array([3, 3]), 5)
    b2 = Bomb(np.array([5, 5]), 5)

    grid = GridWorld([a1, a2], [b1, b2])

    grid.print_state()
    grid.step()
    # print('global reward')
    # print(grid.global_reward)
    grid.print_state()

    print('Iter')
    for i in range(5):
        grid.step()
        grid.print_state()
