import time

import gym
import numpy as np

""" https://bacrobotics.com/Chapter13.html exercise 3"""


class Network:
    def __init__(self, env, nn=(5, 5), p_variance=0.1):
        self.env = env
        n_sensors = env.observation_space.shape[0]
        self.is_box_type = isinstance(env.action_space, gym.spaces.box.Box)
        n_motors = env.action_space.shape[0] if self.is_box_type else env.action_space.n
        self.nn = [n_sensors] + list(nn) + [n_motors]

        self.Ws, self.bs = [], []
        for i in range(len(self.nn) - 1):
            self.Ws.append(np.random.randn(self.nn[i], self.nn[i + 1]) * p_variance)
            self.bs.append(np.zeros(shape=(self.nn[i + 1], 1)))

    def update(self, observation):
        observation.resize(self.nn[0], 1)
        tmp = observation
        for i in range(len(self.nn) - 1):
            tmp = np.dot(self.Ws[i], tmp) + self.bs[i]
            tmp = np.tanh(tmp)
        action = tmp if self.is_box_type else np.argmax(tmp)
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
        n = 0
        for i in range(len(self.nn) - 1):
            n += self.nn[i + 1] * (self.nn[i] + 1)
        return n

    def set_params(self, genotype):
        n = 0
        for i in range(len(self.nn) - 1):
            n1 = n + self.nn[i] * self.nn[i + 1]
            self.Ws[i] = np.resize(genotype[n:n1], (self.nn[i + 1], self.nn[i]))
            n2 = n1 + self.nn[i + 1]
            self.bs[i] = np.resize(genotype[n1:n2], (self.nn[i + 1], 1))
            n += n2


def main(
    gym_env,
    population_size=10,
    nn=(5,),
    gene_std=0.1,
    mut_std=0.02,
    n_episodes=3,
    n_generations=100,
    seed=7777,
):
    np.random.seed(seed)

    env = gym.make(gym_env)
    agent = Network(env, nn)
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
    # main("CartPole-v0")  # works
    main("Pendulum-v1", nn=(10, 10, 10), population_size=100, n_episodes=5, n_generations=20)  # rotates
