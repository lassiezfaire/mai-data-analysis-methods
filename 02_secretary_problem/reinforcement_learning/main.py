from tqdm import tqdm

from reinforcement_learning.agent import SecretaryAgent
from reinforcement_learning.environment import CandidateEnv


def reinforcement_learning(num_candidates: int, n_episodes: int = 200_000) -> list:
    """

    :param num_candidates: Количество кандидатов
    :param n_episodes: Количество эпизодов
    :return: Список рангов кандидатов, которых программа считает лучшими
    """
    env = CandidateEnv(num_candidates=num_candidates)

    # Гиперпараметры
    learning_rate = 0.01
    n_episodes = n_episodes
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = SecretaryAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    ranks_of_chosen = []

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # оборот цикла - эпизод
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # обновляем агента
            agent.update(obs, action, reward, terminated, next_obs)

            # обновляем флаг завершения эпизода и текущее состояние
            done = terminated
            obs = next_obs

        ranks_of_chosen.append(info['rank of current'])
        agent.decay_epsilon()

    return ranks_of_chosen[-int(n_episodes // 2):]
