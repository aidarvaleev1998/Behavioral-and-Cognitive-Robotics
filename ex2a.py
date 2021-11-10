import gym
import numpy as np

""" https://bacrobotics.com/Chapter13.html exercise 3"""


class Network:
    def __init__(self, env, n_hidden=5, p_variance=0.1):
        self.env = env
        self.n_sensors = env.observation_space.shape[0]
        self.is_box_type = isinstance(env.action_space, gym.spaces.box.Box)
        self.n_motors = env.action_space.shape[0] if self.is_box_type else env.action_space.n

        self.W1 = np.random.randn(n_hidden, self.n_sensors) * p_variance
        self.W2 = np.random.randn(self.n_motors, n_hidden) * p_variance
        self.b1 = np.zeros(shape=(n_hidden, 1))
        self.b2 = np.zeros(shape=(self.n_motors, 1))

    def update(self, observation):
        observation.resize(self.n_sensors, 1)
        Z1 = np.dot(self.W1, observation) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = np.tanh(Z2)
        action = A2 if self.is_box_type else np.argmax(A2)
        return action

    def evaluate(self, n_episodes):
        result = []
        for j in range(n_episodes):
            observation = self.env.reset()
            done = False
            fitness = 0
            i = 0

            while not done:
                i += 1
                action = self.update(observation)
                observation, reward, done, info = self.env.step(action)
                fitness += reward

                if done or i >= 200:
                    break

            result.append(fitness)
        return sum(result) / n_episodes


def main():
    env = gym.make("CartPole-v0")
    agent = Network(env, 5)
    print(f"The average fitness over episodes: {agent.evaluate(10)}")
    env.close()


if __name__ == "__main__":
    main()
