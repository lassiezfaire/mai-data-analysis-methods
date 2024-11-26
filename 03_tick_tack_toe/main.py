import numpy as np
import matplotlib.pyplot as plt

from game import reinforcement_learning, play_with_human

from environment import X, O


def plot_learning_curve(reward, window_size = 1000):
    smoothed_reward = np.convolve(reward, np.ones(window_size) / window_size, mode='valid')

    plt.plot(smoothed_reward, label='Скользящее среднее наград')

    plt.xlabel('Эпизоды')
    plt.ylabel('Средний результат')

    plt.title('Кривая обучения модели')
    plt.legend()
    plt.show()

def main():
    # rewards = reinforcement_learning(
    #     size=3,
    #     n_episodes=500_000,
    #     x_q_values_file='', # q_tables//x_q_table.pkl
    #     o_q_values_file='', # q_tables//o_q_table.pkl
    # )
    #
    # plot_learning_curve(rewards, window_size=1000)

    play_with_human(config=[O, 'q_tables//x_q_table.pkl'], size=3)


if __name__ == "__main__":
    main()
