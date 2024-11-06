import random

import gymnasium as gym
import numpy as np


def find_key_by_value(dictionary: dict, value):
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    try:
        return keys[values.index(value)]
    except ValueError:
        return -1


class SecretaryProblemEnv(gym.Env):
    def __init__(self, num_candidates=100):
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


env = SecretaryProblemEnv(10)
observation, info = env.reset()
print(env.candidates)

episode_over = False
while not episode_over:
    action = 0
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"observation: {observation},"
          f"reward: {reward},"
          f"terminated: {terminated},"
          f"truncated: {truncated}",
          f"info: {info}"
          )

    episode_over = terminated
