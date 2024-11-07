from collections import defaultdict
import random
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def find_key_by_value(dictionary: dict, value):
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    try:
        return keys[values.index(value)]
    except ValueError:
        return -1


class CandidateEnv(gym.Env):
    def __init__(self, num_candidates=100):
        """

        :param num_candidates: Количество кандидатов
        """
        self.num_candidates = num_candidates  # количество кандидатов
        self.observation_space = gym.spaces.Discrete(num_candidates)  # наблюдения - это кандидаты
        self.action_space = gym.spaces.Discrete(2)  # 2 действия - принять кандидата или отклонить
        self.reset()

    def candidate_dictionary(self) -> dict:
        # создаём список кандидатов
        random_candidates = set()
        # генерируем уникальных кандидатов до тех пор, пока их не наберётся num_candidates
        while len(random_candidates) < self.num_candidates:
            random_candidates.add(random.randint(10 ** 3, 10 ** 4 - 1))

        random_candidates = list(random_candidates)
        # сортируем кандидатов от лучшего к худшему
        random_candidates.sort(reverse=True)
        # генерируем их ранги
        candidates_ranks = list(range(1, self.num_candidates + 1))
        # объединяем кандидатов и ранги в словарь вида "ранг": "кандидат"
        candidates = {}
        for i in range(self.num_candidates):
            candidates[candidates_ranks[i]] = random_candidates[i]

        # перемешаем кандидатов в случайном порядке с сохранением рангов
        random.shuffle(candidates_ranks)
        shuffled_candidates = {}
        for i in range(self.num_candidates):
            shuffled_candidates[candidates_ranks[i]] = candidates[candidates_ranks[i]]

        return shuffled_candidates

    def _get_obs(self) -> dict:
        return {
            "current candidate": self.current_candidate,
            "best candidate so far": self.best_candidate_so_far,
            "rejected candidates": self.rejected_candidates
        }

    def _get_info(self) -> dict:
        return {
            "rank of current candidate": find_key_by_value(
                dictionary=self.candidates,
                value=self.current_candidate
            ),
            "rank of best candidate so far": find_key_by_value(
                dictionary=self.candidates,
                value=self.best_candidate_so_far
            ),
            "best candidate": self.best_candidate
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.candidates = self.candidate_dictionary()  # список кандидатов
        self.best_candidate = self.candidates[1]  # лучший кандидат (имеющий ранг 1)

        self.current_step = 0  # порядковый номер текущего кандидата в списке кандидатов, обновляется в step()
        self.current_candidate = list(self.candidates.values())[0]  # текущий кандидат
        self.best_candidate_so_far = 0  # лучший из отвергнутых кандидатов (без ранга), обновляется в step()
        self.rejected_candidates = []  # список отвергнутых кандидатов, пополняется в step()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        reward = 0
        self.current_candidate = list(self.candidates.values())[self.current_step]

        if self.best_candidate_so_far < self.current_candidate:
            self.best_candidate_so_far = self.current_candidate

        if action == 1:  # выбираем кандидата
            terminated = True  # останавливаем процесс
            # назначаем высокую награду за выбор лучшего кандидата
            reward = 100 if self.current_candidate == self.best_candidate else -1
        else:  # отвергаем кандидата
            if self.current_step == len(self.candidates.values()) - 1:  # если это последний кандидат - выбираем его
                terminated = True
                reward = 100 if self.current_candidate == self.best_candidate else -1
            else:
                terminated = False
                # добавляем пропущенного кандидата в список пропущенных кандидатов
                self.rejected_candidates.append(self.current_candidate)
                self.current_step += 1

        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        return observation, reward, terminated, truncated, info


class SecretaryAgent:
    def __init__(
            self,
            env: CandidateEnv,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: Dict[str, Any]) -> int:
        hashible_obs = tuple(obs.values())[:2]

        if np.random.random() < self.epsilon:
            return 0 if np.random.random() < 20 / 21 else 1
        else:
            return int(np.argmax(self.q_values[hashible_obs]))

    def update(
            self,
            obs: Dict[str, Any],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: Dict[str, Any],
    ):

        hashible_obs = tuple(obs.values())[:2]
        hashable_next_obs = tuple(next_obs.values())[:2]

        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[hashable_next_obs])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[hashible_obs][action]
        )

        self.q_values[hashible_obs][action] = (
                self.q_values[hashible_obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


env = CandidateEnv(100)
observation, info = env.reset()

# hyperparameters
learning_rate = 0.01
n_episodes = 200_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = SecretaryAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

first_rank = 0
ranks_by_episode = {}

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated
        obs = next_obs

    if info['rank of current candidate'] == 1:
        first_rank += 1

    if episode % 10000 == 0:
        ranks_by_episode[episode] = first_rank
    agent.decay_epsilon()

print(ranks_by_episode)
# plt.figure(figsize=(10, 6))
# plt.hist(ranks, bins=100)
# plt.xticks(np.arange(0, 101, 10))
# plt.ylim(0, 40000)
# plt.xlabel('Chosen candidate')
# plt.ylabel('frequency')
plt.show()


# print(env.candidates)
#
# episode_over = False
# while not episode_over:
#     action = 0
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(f"observation: {observation}, "
#           f"reward: {reward}, "
#           f"terminated: {terminated}, "
#           f"truncated: {truncated}, "
#           f"info: {info}"
#           )
#
#     episode_over = terminated
