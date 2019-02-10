import gym

env = gym.make("Pendulum-v0")

env.reset()
env.render()

while True:
    action = env.action_space.sample()
    print(action)
    env.step(action)
    env.render()