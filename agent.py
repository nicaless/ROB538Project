import numpy as np


ROWS = 50
COLS = 50

class Agent:
    def __init__(self, init_pos, type_name, defusal_types, defusal_skill, mobility, sensing, eps, bounds=(ROWS, COLS)):
        """

        :param init_pos: agent's initial ability
        :param defusal_skill: parameter for agent's skill
        :param mobility: parameter for how far agent can move in a time step
        :param sensing: parameter for how far agent can sense in a time step
        :param eps: parameter for epsilon greedy
        """
        self.position = init_pos
        self.type_name = type_name
        self.defusal_skill = defusal_skill
        self.mobility = mobility
        self.sensing = sensing
        self.eps = eps
        self.ROWS = bounds[0]
        self.COLS = bounds[1]

        # action values:  0) move to/stay at bomb OR 1) move away
        self.action_values = [[0], [0]]
        self.failed = False  # flag for if agent is still available
        self.defusal_types = defusal_types  # dictionary to keep track of the defusal skills of each type
        self.team_config = {}  # shared information about the team's current configuration
        self.add_types = {k: 0 for k in self.defusal_types.keys()}

    def get_team_config(self, team_config):
        # update knowledge about the team configuration
        self.team_config = team_config

    def move(self, target=None):
        # print('Moving')
        if target is not None:
            # print('Currently At')
            # print(self.position)
            x_diff = target[0] - self.position[0]
            y_diff = target[1] - self.position[1]
            # print(x_diff, y_diff)

            x_inc = np.min([np.abs(x_diff), self.mobility])
            y_inc = np.min([np.abs(y_diff), self.mobility])
            # print(x_inc, y_inc)

            self.position[0] = self.position[0] + np.sign(x_diff) * x_inc
            self.position[1] = self.position[1] + np.sign(y_diff) * y_inc
            # print('Now At')
            # print(self.position)

        else:
            self.move_random()

    def move_random(self):
        """
        only moves in random cardinal direction
        :return:
        """
        dir = np.random.choice(['LEFT', 'RIGHT', 'UP', 'DOWN', 'STAY'])
        inc = np.random.choice(list(range(1,self.mobility+1)))
        # print("Move Random", dir, inc)

        curr_pos = self.position
        curr_x = curr_pos[0]
        curr_y = curr_pos[1]

        if dir == 'LEFT':
            new_y = max(curr_y-inc, 0)
            new_pos = np.array([curr_x, new_y])
        elif dir == 'RIGHT':
            new_y = min(curr_y+inc, self.COLS-1)
            new_pos = np.array([curr_x, new_y])
        elif dir == 'UP':
            new_x = max(curr_x-inc, 0)
            new_pos = np.array([new_x, curr_y])
        elif dir == 'DOWN':
            new_x = min(curr_x+inc, self.ROWS-1)
            new_pos = np.array([new_x, curr_y])
        else:
            new_pos = curr_pos

        self.position = new_pos


    def sense(self, grid):
        """
        return closest bomb

        :param grid:
        :return:
        """
        curr_pos = self.position

        x_range_low = max(curr_pos[0]-self.sensing, 0)
        x_range_high = min(curr_pos[0]+self.sensing, self.ROWS-1)
        y_range_low = max(curr_pos[1] - self.sensing, 0)
        y_range_high = min(curr_pos[1] + self.sensing, self.COLS-1)

        bombs = []
        for x in range(x_range_low, x_range_high+1):
            for y in range(y_range_low, y_range_high+1):
                if grid[x, y] > 1:
                    bombs.append(np.array([x, y]))

        if len(bombs) == 0:
            return None

        # print(bombs)
        # print(np.linalg.norm(curr_pos - bombs, axis=1))
        # print(np.argmin(np.linalg.norm(curr_pos - bombs, axis=1)))
        bomb_loc = np.argmin(np.linalg.norm(curr_pos - bombs, axis=1))
        # print(bombs[bomb_loc])
        return bombs[bomb_loc]

    def act(self, target):
        # if no target move randomly
        # if target is present, move towards target
        # if at target, decide whether to stay or go according to reward (or epsilon)
        if target is not None:
            if np.array_equal(target, self.position):
                p = np.random.random()
                if p < self.eps:
                    j = np.random.choice(2)
                else:
                    j = np.argmax([np.mean(rewards) for rewards in self.action_values])

                if j == 1:  # move away from bomb
                    # TODO: should we give some really small reward for moving away (rewarding self-preservation?)
                    self.action_values[0].append(0.001)
                    self.move_random()
                    pass
        self.move(target)

    def update_probabilities(self, N, global_reward, bomb_states, bomb_skill_level):
        reward = self.reward(N, global_reward, bomb_states, bomb_skill_level)
        self.action_values[0].append(reward)
        # print('Agent Received Reward')
        # print(reward)

    def reward(self, N, global_reward, bomb_states, bomb_skill_level):
        # calculate D
        defused_with_me = False
        defused_without_me = False
        agent_skills = []
        for bomb_position, agents_at_bomb in bomb_states.items():
            if np.array_equal(self.position, bomb_position):
                agent_skills = [a.defusal_skill for a in agents_at_bomb]
                total_agent_skill = np.sum(agent_skills)
                if total_agent_skill >= bomb_skill_level[bomb_position]:
                    defused_with_me = True
                else:
                    defused_with_me = False

                cf_agents_at_bomb = []
                for agent in agents_at_bomb:
                    if np.array_equal(self.position, agent.position):
                       continue
                    cf_agents_at_bomb.append(agent)
                cf_agent_skill = np.sum([a.defusal_skill for a in cf_agents_at_bomb])

                if cf_agent_skill >= bomb_skill_level[bomb_position]:
                    defused_without_me = True
                else:
                    defused_without_me = False
                break

        # Not at bomb POI, do not receive any reward yet
        if len(agent_skills) == 0:
            return 0

        # if bomb was defused and could not have been defused without me, receive a difference reward
        if (not defused_without_me) and defused_with_me:
            D = 1
        # if bomb was not defused with or without me, or was defused without my help, receive no difference reward
        else:
            D = 0

        # calculate D++
        dplusplus = self.dplusplus_reward(N, global_reward, bomb_states, bomb_skill_level)
        # if adding more of me does not produce any benefit, just use D
        if dplusplus <= D:
            # print('Using Difference Reward {}'.format(D))
            return D

        # print('D++ Loop', len(agent_skills)+1, N)
        dplusplus_prev_n = D
        for i in range(len(agent_skills)+1, N):
            dplusplus_n = self.dplusplus_reward(i, global_reward, bomb_states, bomb_skill_level, sample_agents=True)
            if dplusplus_n > dplusplus_prev_n:
                # print('Using D++ Reward {}'.format(D))
                return dplusplus_n
        # print('Using Difference Reward {}'.format(D))
        return D

    def dplusplus_reward(self, N, global_reward, bomb_states, bomb_skill_level, sample_agents=False):
        defused_with_cf = False
        defused_without_cf = False
        config = dict(self.team_config)
        for bomb_position, agents_at_bomb in bomb_states.items():
            if np.array_equal(self.position, bomb_position):
                agent_skills = [a.defusal_skill for a in agents_at_bomb]
                total_agent_skill = np.sum(agent_skills)
                cf_agents_skill = []
                for i in range(len(agents_at_bomb), N):
                    if sample_agents:
                        cf = self.type_name
                        cf_skill = np.random.choice(agent_skills)
                        # get the type name for the skill level used in counterfactual
                        for cf_type, skill in self.defusal_types.items():
                            if skill == cf_skill:
                                cf = cf_type
                                break
                        config[cf] -= 1
                    else:
                        cf_skill = self.defusal_skill
                    cf_agents_skill.append(cf_skill)


                total_cf_agents_skill = total_agent_skill + np.sum(cf_agents_skill)

                if total_agent_skill >= bomb_skill_level[bomb_position]:
                    defused_without_cf = True
                else:
                    defused_without_cf = False

                if total_cf_agents_skill >= bomb_skill_level[bomb_position]:
                    defused_with_cf = True
                else:
                    defused_with_cf = False

        # if bomb could be defused with more of me and not otherwise, d++ reward is positive
        if defused_with_cf and (not defused_without_cf):
            D = global_reward + 1

            # Check if we added more agents of a certain type than available
            for atype, kn in config.items():
                if kn < 0:
                    self.add_types[atype] += 1

        # if bomb cannot be defused no matter how many more of me, receive no d++ reward
        else:
            D = 0

        # print('D++ reward {}'.format(N))
        # print(D / (N-1))
        return D / (N-1)

    def step(self, grid):
        target = self.sense(grid)
        # print('See Targets')
        # print(target)
        self.act(target)

