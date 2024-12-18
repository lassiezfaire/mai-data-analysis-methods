from typing import List, Tuple, Union

from tqdm import tqdm

from bot import BotPlayer
from human import HumanPlayer
from environment import BoardEnv, X

def bot_turn(bot: BotPlayer, obs: tuple, env: BoardEnv) -> Tuple[bool, bool, tuple, dict, int]:
    """Один ход бота (включая попытки походить на уже занятые клетки)

    :return: Кортеж из полезной для обучения информации
    """

    episode_reward = 0
    done = False

    while not done:
        action = bot.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

        done = True if reward >= -1 else False
        bot.update(obs, action, reward, terminated or truncated, next_obs)

    return terminated, truncated, next_obs, info, episode_reward

def reinforcement_learning(
        size: int,
        n_episodes: int,
        x_q_values_file='q_tables//x_q_table.pkl',
        o_q_values_file='q_tables//o_q_table.pkl'
) -> list:
    """Функция обучения с подкреплением

    :param size: размер игрового поля
    :param n_episodes: количество эпизодов обучения
    :param x_q_values_file: путь к файлу сохранения q-таблицы для игрока X
    :param o_q_values_file: путь к файлу сохранения q-таблицы для игрока O
    :return: награды за каждый эпизод обучения
    """

    env = BoardEnv(size=size)

    # Гиперпараметры
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)

    x_agent = BotPlayer(
        env=env,
        epsilon_decay=epsilon_decay,
        q_values_file=x_q_values_file
    )

    o_agent = BotPlayer(
        env=env,
        epsilon_decay=epsilon_decay,
        q_values_file=o_q_values_file
    )

    rewards = []

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()

        done = False

        episode_reward = 0

        # оборот цикла - эпизод
        while not done:

            # Ход крестик
            terminated, truncated, next_obs, info, episode_reward = bot_turn(bot=x_agent, obs=obs, env=env)

            # обновляем флаг завершения эпизода и текущее состояние
            done = terminated or truncated
            obs = next_obs

            if done:
                break

            # Ход нолик
            terminated, truncated, next_obs, info, episode_reward = bot_turn(bot=o_agent, obs=obs, env=env)

            # обновляем флаг завершения эпизода и текущее состояние
            done = terminated or truncated
            obs = next_obs

        rewards.append(episode_reward)

        x_agent.decay_epsilon()
        o_agent.decay_epsilon()

    # print(rewards)

    x_agent.save_q_table(filename='q_tables//x_q_table.pkl')
    o_agent.save_q_table(filename='q_tables//o_q_table.pkl')

    return rewards

def play_with_human(config: List[Union[int, str]], size: int):
    """Функция игры человека и машины

    :param config: сторона, за которую играет игрок, и путь к файлу с Q-таблицей для бота
    :param size: размер игрового поля
    :return:
    """

    side, q_values_file = config[0], config[1]

    env = BoardEnv(size=size)

    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (1 / 2)

    human = HumanPlayer(
        size=size
    )

    bot = BotPlayer(
        env=env,
        epsilon_decay=epsilon_decay,
        q_values_file=q_values_file
    )

    print(f'Добро пожаловать в игру "Крестики-нолики".')

    obs, info = env.reset()

    env.print_diagnostic(_get_info_dict=info, terminated=False, truncated=False)

    done = False

    while not done:
        if side == X:
            action = human.get_action()
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            obs = next_obs

            env.print_diagnostic(_get_info_dict=info, terminated=terminated, truncated=truncated)

            if done:
                break

            terminated, truncated, next_obs, info, episode_reward = bot_turn(bot=bot, obs=obs, env=env)

            done = terminated or truncated
            obs = next_obs


            env.print_diagnostic(_get_info_dict=info, terminated=terminated, truncated=truncated)

        else:
            terminated, truncated, next_obs, info, episode_reward = bot_turn(bot=bot, obs=obs, env=env)

            done = terminated or truncated
            obs = next_obs

            print(terminated, truncated)
            env.print_diagnostic(_get_info_dict=info, terminated=terminated, truncated=truncated)

            if done:
                break

            action = human.get_action()
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            obs = next_obs

            print(terminated, truncated)
            env.print_diagnostic(_get_info_dict=info, terminated=terminated, truncated=truncated)

    bot.decay_epsilon()
    print('Игра окончена.')
