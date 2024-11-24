from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from agent import PlayerAgent
from environment import BoardEnv


def plot_learning_curve(reward, window_size=1000):
    smoothed_reward = np.convolve(reward, np.ones(window_size) / window_size, mode='valid')

    plt.plot(smoothed_reward, label='Скользящее среднее наград')

    plt.xlabel('Эпизоды')
    plt.ylabel('Средний результат')

    plt.title('Кривая обучения модели')
    plt.legend()
    plt.show()


def reinforcement_learning(size: int, n_episodes: int = 100_000) -> list:
    env = BoardEnv(size=size)

    # Гиперпараметры
    learning_rate = 0.01
    n_episodes = n_episodes
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = PlayerAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    rewards = []
    turns = []

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        episode_reward = 0

        # оборот цикла - эпизод
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # обновляем агента
            agent.update(obs, action, reward, terminated, next_obs)

            # обновляем флаг завершения эпизода и текущее состояние
            done = terminated
            obs = next_obs

        rewards.append(episode_reward)
        turns.append(info['current_turn'])
        # env.print_diagnostic(_get_info_dict=info, terminated=terminated)

        agent.decay_epsilon()

    # print(turns)
    # print(rewards)

    agent.save_q_table()
    plot_learning_curve(rewards, window_size=100)

def main():
    reinforcement_learning(size=3, n_episodes=20_000)


if __name__ == "__main__":
    main()
