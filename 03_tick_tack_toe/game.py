from typing import List

from tqdm import tqdm

from bot import BotPlayer
from human import HumanPlayer
from environment import BoardEnv, X

def bot_turn(bot: BotPlayer, obs: tuple, env: BoardEnv):
    """

    :return:
    """

    o_step_done = False

    while not o_step_done:
        action = bot.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        o_step_done = True if reward >= -1 else False
        bot.update(obs, action, reward, terminated, next_obs)

    return terminated, next_obs, info

def reinforcement_learning(
        size: int,
        n_episodes: int,
        x_q_values_file='q_tables//x_q_table.pkl',
        o_q_values_file='q_tables//o_q_table.pkl'
) -> list:
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
            x_step_done = False
            while not x_step_done:
                if type(x_agent) == BotPlayer:
                    action = x_agent.get_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    x_step_done = True if reward >= -1 else False
                    x_agent.update(obs, action, reward, terminated, next_obs)

            # обновляем флаг завершения эпизода и текущее состояние
            done = terminated
            obs = next_obs

            if done:
                break

            # Ход нолик
            o_step_done = False
            while not o_step_done:
                action = o_agent.get_action(obs) if type(o_agent) == BotPlayer else o_agent.get_action()
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                o_step_done = True if reward >= -1 else False
                o_agent.update(obs, action, reward, terminated, next_obs)

            # обновляем флаг завершения эпизода и текущее состояние
            done = terminated
            obs = next_obs

            # env.print_diagnostic(_get_info_dict=info, terminated=terminated)

        rewards.append(episode_reward)

        x_agent.decay_epsilon()
        o_agent.decay_epsilon()

    # print(rewards)

    x_agent.save_q_table(filename='q_tables//x_q_table.pkl')
    o_agent.save_q_table(filename='q_tables//o_q_table.pkl')

    return rewards

def play_with_human(config: List, size: int):
    """

    :param config:
    :param size:
    :param n_games:
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

    env.print_diagnostic(_get_info_dict=info, terminated=False)

    done = False

    while not done:
        if side == X:
            action = human.get_action()
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated
            obs = next_obs

            env.print_diagnostic(_get_info_dict=info, terminated=terminated)

            if done:
                break

            terminated, next_obs, info = bot_turn(bot=bot, obs=obs, env=env)

            done = terminated
            obs = next_obs

            env.print_diagnostic(_get_info_dict=info, terminated=terminated)
        else:
            terminated, next_obs, info = bot_turn(bot=bot, obs=obs, env=env)

            done = terminated
            obs = next_obs

            env.print_diagnostic(_get_info_dict=info, terminated=terminated)

            if done:
                break

            action = human.get_action()
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated
            obs = next_obs

            env.print_diagnostic(_get_info_dict=info, terminated=terminated)

    bot.decay_epsilon()
    print('Игра окончена.')
