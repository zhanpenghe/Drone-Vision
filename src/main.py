from environment import DroneEnv as Env
from reinforcement_learning import DeepQNetwork


def run_rl(env, rl):

    for episode in range(10000):
        observation = env.reset()
        while True:
            action = rl.choose_action(observation)
            observation_, reward, done = env.step(action)
            rl.store_transition(observation, action, reward, observation_, done)
            rl.learn()
            observation = observation_
            if done:
                break


def main():
    env = Env()
    RL = DeepQNetwork(n_actions=100,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      epsilon_greedy=0.9,
                      memory_size=2000)
    run_rl(env, RL)


if __name__ == '__main__':
    main()