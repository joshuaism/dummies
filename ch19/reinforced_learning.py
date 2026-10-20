import numpy as np
import matplotlib.pyplot as plt


class LineWorld:
    """Markov chain with states in a line from 0 to 4 with 2 as the starting state.
    Actors can move up or down the line by one with a reward at state 4 and a penalty 
    at state 0"""

    def __init__(self):
        self.start_state = 2
        self.state = self.start_state
        self.n_states = 5
        self.n_actions = 2

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        if action == 0:
            self.state -= 1
        else:
            self.state += 1

        self.state = max(0, min(self.state, self.n_states - 1))
        done = self.state in [0, 4]

        if self.state == 4:
            reward = 1.0
        elif self.state == 0:
            reward = -1
        else:
            reward = 0

        return self.state, reward, done


class PolicyGradientAgent:
    """implementation of an episode-learning policy gradient agent algorithm"""

    def __init__(self, n_states, n_actions, learning_rate=0.05, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.preferences = np.zeros((n_states, n_actions))

    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def get_action_probs(self, state):
        return self.softmax(self.preferences[state])

    def select_action(self, state):
        probs = self.get_action_probs(state)
        action = np.random.choice(self.n_actions, p=probs)
        return action

    def compute_returns(self, rewards):
        returns = np.zeros(len(rewards))
        G = 0.0

        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        return returns

    def update_episode(self, states, actions, rewards):
        returns = self.compute_returns(rewards)

        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

            for state, action, G in zip(states, actions, returns):
                probs = self.get_action_probs(state)

                grad_log_pi = -probs.copy()
                grad_log_pi[action] += 1.0

                self.preferences[state] += self.lr * G * grad_log_pi


def train(episodes=500):
    env = LineWorld()
    agent = PolicyGradientAgent(env.n_states, env.n_actions)
    rewards_per_episode = []
    prob_right = []

    for ep in range(episodes):
        state = env.reset()

        states = []
        actions = []
        rewards = []

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        agent.update_episode(states, actions, rewards)

        total_reward = sum(rewards)
        rewards_per_episode.append(total_reward)
        prob_right.append(agent.get_action_probs(2)[1])

        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(
                f"Episode {ep + 1} | Avg Reward: {avg_reward:.3f} | P(Right|2): {prob_right[-1]:.3f}"
            )

    return agent, rewards_per_episode, prob_right


def plot_results(rewards, prob_right):
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
    else:
        moving_avg = rewards

    plt.figure()
    plt.plot(rewards, alpha=0.3, label="Raw Reward")
    if len(rewards) >= window:
        plt.plot(
            range(window - 1, len(rewards)),
            moving_avg, label="Smoothed Reward"
        )
    else:
        plt.plot(moving_avg, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(prob_right)
    plt.xlabel("Episode")
    plt.ylabel("P(Right)")
    plt.title("Policy Learning at Start State")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    agent, rewards, prob_right = train()

    print("\nFinal Policy:")
    for s in range(agent.n_states):
        probs = agent.get_action_probs(s)
        print(f"State{s}:Left={probs[0]:.2f}, Right={probs[1]:.2f}")

    plot_results(rewards, prob_right)
