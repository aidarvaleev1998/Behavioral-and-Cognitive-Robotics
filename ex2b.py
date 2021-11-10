import time

import gym
import numpy as np

""" https://bacrobotics.com/Chapter13.html exercise 3"""


class Network:
    def __init__(self, env, n_hidden=5, p_variance=0.1):
        self.env = env
        self.n_hidden = n_hidden
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

    def evaluate(self, n_episodes, render=False):
        result = []
        for j in range(n_episodes):
            observation = self.env.reset()
            done = False
            fitness = 0
            i = 0

            while not done:
                if render:
                    self.env.render()
                    time.sleep(0.1)

                i += 1
                action = self.update(observation)
                observation, reward, done, info = self.env.step(action)
                fitness += reward

                if done or i >= 200:
                    if render:
                        time.sleep(3)
                    break

            result.append(fitness)
        return sum(result) / n_episodes

    def get_n_params(self):
        return self.n_hidden * (self.n_sensors + 1) + self.n_motors * (self.n_hidden + 1)

    def set_params(self, genotype):
        n1 = self.n_hidden * self.n_sensors
        self.W1 = np.resize(genotype[:n1], (self.n_hidden, self.n_sensors))
        n2 = n1 + self.n_motors * self.n_hidden
        self.W2 = np.resize(genotype[n1:n2], (self.n_motors, self.n_hidden))
        n3 = n2 + self.n_hidden
        self.b1 = np.resize(genotype[n2:n3], (self.n_hidden, 1))
        n4 = n3 + self.n_motors
        self.b2 = np.resize(genotype[n3:n4], (self.n_motors, 1))


def main(
    gym_env,
    population_size=10,
    n_hidden=20,
    gene_std=0.1,
    mut_std=0.02,
    n_episodes=3,
    n_generations=100,
    seed=777,
):
    np.random.seed(seed)

    env = gym.make(gym_env)
    agent = Network(env, n_hidden)
    n_params = agent.get_n_params()
    population = np.random.randn(population_size, n_params) * gene_std

    for g in range(n_generations):
        fitness = np.zeros(population_size)
        for i in range(population_size):
            agent.set_params(population[i])
            fitness[i] = agent.evaluate(n_episodes)
        idx = np.argsort(-fitness)
        fitness = fitness[idx]
        population = population[idx]
        for i in range(population_size // 2):
            population[i + population_size // 2] = population[i] + np.random.randn(n_params) * mut_std
        print(f"{g}: max fitness {fitness.max():.1f}, avg fitness {fitness.mean():.1f}")

    agent.set_params(population[0])
    print(f"The average fitness over episodes of the fittest: {agent.evaluate(10)}")
    agent.evaluate(1, True)
    env.close()


if __name__ == "__main__":
    main("CartPole-v0")  # works
    # main("Pendulum-v1", population_size=100, n_episodes=5, n_generations=100, mut_std=0.05, n_hidden=20)  # rotates
