# Q-learning with e-greedy
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent():
    def __init__(self, agent_name, agent_type, actions=[1, 2], epsilon_init=1.0, epsilon_threshold=0.0, epsilon_decay=0.01, alpha=0.01, gamma=0.95):

        self.name = agent_name
        self.type = agent_type # "X" or "Y"
        self.action_space = actions

        # Q-table init - use dict to simplify action to index converesion
        self.q_table = {action: 0.0 for action in actions}

        # Hyperparams
        self.epsilon = epsilon_init
        self.epsilon_threshold = epsilon_threshold
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

        # Metric tracking
        self.reward_history = []
        self.action_value_history = {1: [], 2: []}

    def action(self):
        if np.random.rand() < self.epsilon:
            chosen_action = np.random.choice(self.action_space)
        else:
            max_val = max(self.q_table.values())
            best_action = [a for a, v in self.q_table.items() if v == max_val]
            chosen_action = np.random.choice(best_action)

        return chosen_action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_threshold:
            self.epsilon -= self.epsilon_decay

    def update_q_values(self, action, reward):
        # Standard Q-learning update: Q(a) = Q(a) + alpha * (reward + gamma * max(Q) - Q(a))
        max_next_q = max(self.q_table.values())
        self.q_table[action] += self.alpha * (reward + (self.gamma * max_next_q) - self.q_table[action])


class environment:
    def __init__(self):
        # [[(Row_Payoff, Col_Payoff), ...], ...]
        self.payoff_matrix = [
            [(1, 2), (1, 1)],
            [(1,1), (2,1)]
        ]

      # Define types: X (Column players) and Y (Row players)
        self.agent_types = {
            "X1": "X", "X2": "X", "X3": "X", "X4": "X", "X5": "X",
            "Y1": "Y", "Y2": "Y"
        }

        # Define the djacency graph edges
        self.graph = {
            "X1": ["Y1"],
            "X2": ["Y2"],
            "Y1": ["X1", "X3", "X4"],
            "Y2": ["X2", "X3", "X5"],
            "X3": ["Y1", "Y2", "X4", "X5"],
            "X4": ["Y1", "X3"],
            "X5": ["Y2", "X3"]
        }

        # Epsilon decay parameters

    def reset(self):
        pass

    def step(self, action_matrix):
        """
        Distributes rewards.
        action_matrix = {agent_name: agent_action}
        """

        rewards = {name: 0 for name in self.graph.keys()}

        # Traverse each agent in the graph
        for each_agent in self.graph.keys():
            # Traverse each opponents and calculate rewards
            for each_opponent in self.graph[each_agent]:
                # Extract actions
                act_agent = action_matrix[each_agent] - 1
                act_opp = action_matrix[each_opponent] - 1

                # Extract row column combo from payoff matrix
                if "Y" in each_agent:
                    row_idx, col_idx = act_agent, act_opp

                elif "Y" in each_opponent:
                    row_idx, col_idx = act_opp, act_agent

                else:
                    # Same pair of agents: X vs X
                    row_idx, col_idx = act_agent, act_opp

                # Extract payoff block
                payoff_block = self.payoff_matrix[row_idx][col_idx]

                if "Y" in each_agent:
                    rewards[each_agent] += payoff_block[0]
                else:
                    rewards[each_agent] += payoff_block[1]

        return rewards


def simulate(total_episodes):

    # Init metrics
    # Init value tracking
    q_table_values = dict()
    q_table_values["Y1"] = [{1: 0.0, 2: 0.0}]
    q_table_values["X1"] = [{1: 0.0, 2: 0.0}]
    q_table_values["X3"] = [{1: 0.0, 2: 0.0}]

    agent_types = {
        "X1": "X", "X2": "X", "X3": "X", "X4": "X", "X5": "X",
        "Y1": "Y", "Y2": "Y"
    }

    # Init enviromnet
    env = environment()

    # Init agents
    # Define types: X (Column players) and Y (Row players)
    agents = dict()
    for name, agent_type in agent_types.items():
        agents[name] = QLearningAgent(name, agent_type)

    # Reset env
    env.reset()

    # Simulation start
    for episode in range(total_episodes):
        current_actions = {}

        # Register actions
        for name, agent in agents.items():
            current_actions[name] = agent.action()

        # Step the environment
        rewards = env.step(current_actions)

        # Update agent strategies
        for each_agent in rewards.keys():
            agents[each_agent].update_q_values(current_actions[each_agent], rewards[each_agent])

        # Decay epsilon
        if (episode > 0) and (episode % 40 == 0):
            for agent in agents.values():
                agent.decay_epsilon()

        # Accmulate value metrics
        q_table_values["Y1"].append(dict(agents["Y1"].q_table))
        q_table_values["X1"].append(dict(agents["X1"].q_table))
        q_table_values["X3"].append(dict(agents["X3"].q_table))

    # Plot q-values for Y1,X1,X3
    # For each agent, we need 2 datapoints for the actions

    # Y1
    y1_action1_values = [entry[1] for entry in q_table_values["Y1"]]
    y1_action2_values = [entry[2] for entry in q_table_values["Y1"]]
    # plt.plot(range(len(y1_action1_values)), y1_action1_values, label="Y1 Action1", linestyle="solid")
    # plt.plot(range(len(y1_action2_values)), y1_action2_values, label="Y1 Action2", linestyle="dotted")

    #X1
    x1_action1_values = [entry[1] for entry in q_table_values["X1"]]
    x1_action2_values = [entry[2] for entry in q_table_values["X1"]]
    # plt.plot(range(len(x1_action1_values)), x1_action1_values, label="X1 Action1", linestyle="dashed")
    # plt.plot(range(len(y1_action2_values)), x1_action2_values, label="X1 Action2", linestyle="dashdot")

    #X3
    x3_action1_values = [entry[1] for entry in q_table_values["X3"]]
    x3_action2_values = [entry[2] for entry in q_table_values["X3"]]
    # plt.plot(range(len(x3_action1_values)), x3_action1_values, label="X3 Action1")
    # plt.plot(range(len(x3_action2_values)), x3_action2_values, label="X3 Action2", linestyle="-")

    # Plotting
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(range(len(y1_action1_values)), y1_action1_values, label="Y1 Action1", linestyle="solid")
    axs[0].plot(range(len(y1_action2_values)), y1_action2_values, label="Y1 Action2", linestyle="dotted")
    # axs[0].set(xlabel="Episode")
    axs[0].legend(loc='best')
    axs[0].set(ylabel="Q-Values")
    axs[0].set_title("Y1 Action values")

    axs[1].plot(range(len(x1_action1_values)), x1_action1_values, label="X1 Action1", linestyle="dashed")
    axs[1].plot(range(len(y1_action2_values)), x1_action2_values, label="X1 Action2", linestyle="dashdot")
    # axs[1].set(xlabel="Episode")
    axs[1].legend(loc='best')
    axs[1].set(ylabel="Q-Values")
    axs[1].set_title("X1 Action values")

    axs[2].plot(range(len(x3_action1_values)), x3_action1_values, label="X3 Action1")
    axs[2].plot(range(len(x3_action2_values)), x3_action2_values, label="X3 Action2", linestyle="-")
    axs[2].set(xlabel="Episode")
    axs[2].legend(loc='best')
    axs[2].set(ylabel="Q-Values")
    axs[2].set_title("X3 Action values")

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate(5000)
