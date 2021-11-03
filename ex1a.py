import gym
import time

env = gym.make("MountainCar-v0")
observation = env.reset()
done = False
fitness = 0
i = 0

while not done:
    i += 1
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    fitness += reward

    print(f"\nStep: {i}\nObservation vector: {observation}\n"
          f"Action vector: {action}\nReward: {reward}\nFitness: {fitness}")

    if done:
        time.sleep(3)

    time.sleep(0.1)

env.close()
