import numpy as np

from agent import Agent

ROWS = 10
COLS = 10


class Bomb:
    def __init__(self, position, skill_level):
        self.position = position
        self.skill_level = skill_level
        self.defused = False


class GridWorld:
    def __init__(self, agents, bombs):
        self.agents = agents
        self.bombs = bombs
        self.grid_state =  np.zeros((ROWS, COLS))
        self.update_state()
        self.global_reward = 0

    def print_state(self):
        print(self.grid_state)
        print('Bombs Defused: {}'.format(self.global_reward))

    def step(self):
        # Move Agents
        for agent in self.agents:
            agent.step(self.grid_state)

        self.update_state()

        # Defuse Any Bombs
        agents_available_to_defuse = []
        bomb_states = {}
        bomb_skill_level = {}
        for bomb in self.bombs:
            agents_at_bomb = []
            for agent in self.agents:
                if np.array_equal(agent.position, bomb.position):
                    agents_at_bomb.append(agent)
                    agents_available_to_defuse.append(agent)
            total_agent_skill = np.sum([agent.defusal_skill for agent in agents_at_bomb])
            if total_agent_skill >= bomb.skill_level:
                bomb.defused = True
                self.global_reward += 1
            bomb_states[(bomb.position[0], bomb.position[1])] = agents_at_bomb
            bomb_skill_level[(bomb.position[0], bomb.position[1])] = bomb.skill_level

        agents_available_to_defuse = set(agents_available_to_defuse)
        for i, agent in enumerate(self.agents):
            # print('Agent {}'.format(i))
            agent.update_probabilities(len(self.agents), self.global_reward,
                                       bomb_states, bomb_skill_level)

        self.update_state()


    def update_state(self):
        grid = np.zeros((ROWS, COLS))

        for agent in self.agents:
            pos = agent.position
            grid[pos[0], pos[1]] = 1

        for bomb in self.bombs:
            if not bomb.defused:
                pos = bomb.position
                grid[pos[0], pos[1]] = 2

        self.grid_state = grid



a1 = Agent(np.array([1, 1]), 3, 1, 5, 0.9)
a2 = Agent(np.array([0, 0]), 3, 1, 5, 0.9)
b1 = Bomb(np.array([2, 2]), 5)
b2 = Bomb(np.array([3, 3]), 5)

grid = GridWorld([a1, a2], [b1, b2])

grid.print_state()
grid.step()
# print('global reward')
# print(grid.global_reward)
grid.print_state()


for i in range(5):
    grid.step()
    grid.print_state()
