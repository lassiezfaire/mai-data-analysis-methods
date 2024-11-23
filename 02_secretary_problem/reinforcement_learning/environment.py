import numpy as np
from gymnasium import Env, spaces
from typing import Tuple

from candidates_generator import generate_candidate_dict


def find_key_by_value(dictionary: dict, value):
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    try:
        return keys[values.index(value)]
    except ValueError:
        return -1


class CandidateEnv(Env):
    """Среда обучения - мир (кандидаты в количестве num_candidates штук), в котором будет действовать агент."""

    def __init__(self, num_candidates: int):
        """
        :param num_candidates: Количество кандидатов
        """

        self.num_candidates = num_candidates  # количество кандидатов
        # количество наблюдений совпадает с количеством кандидатов
        self.observation_space = spaces.Discrete(num_candidates)
        self.action_space = spaces.Discrete(2)  # два действия - принять кандидата или отклонить
        self.reset()

    def _get_obs(self) -> dict:
        """Преобразуем состояние среды (environment) в наблюдение (observation).
        Эти данные подаются на обучение нейронной сети

        :return: словарь-наблюдение
        """
        return {
            # проверка, является ли текущий кандидат лучшим
            'if candidate is best so far': self.candidates[self.current_step] >= self.best_candidate_so_far,
            'share of rejected': int(self.current_step / self.num_candidates * 100)
        }

    def _get_info(self) -> dict:
        """Диагностические данные, которые позволяют отслеживать работу нейронной сети

        :return: словарь диагностической информации
        """
        return {
            'rank of current': find_key_by_value(
                dictionary=self.candidates_dict,
                value=self.current_candidate
            )
        }

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        """Инициализирует новый эпизод для среды, случайным образом выбирая позиции агента и цели.

        :param seed: Инициализирует генератор случайных чисел для детерминированного состояния.
        :param options: Дополнительные параметры для настройки сброса.
        :return: Кортеж из начального наблюдения и дополнительной информации.
        """

        super().reset(seed=seed)
        self.candidates_dict = generate_candidate_dict(num_candidates=self.num_candidates)
        self.candidates = list(self.candidates_dict.values())
        self.candidates = np.asarray(self.candidates)

        self.ranks = list(self.candidates_dict.keys())
        self.ranks = np.asarray([int(i - 1) for i in self.ranks])

        self.current_step = 0  # порядковый номер текущего кандидата в списке кандидатов
        self.current_candidate = self.candidates[self.current_step]  # текущий кандидат

        self.best_candidate_so_far = 0  # качество лучшего из просмотренных кандидата

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action) -> Tuple[dict, int, bool, bool, dict]:
        """Выполняет действие, вычисляет новое состояние среды и возвращает кортеж из следующего наблюдения,
        вознаграждения, признаков завершения и дополнительной информации.

        :param action: Действие, выполняемое агентом.
        :return: Кортеж из следующего наблюдения, вознаграждения, признаков завершения и дополнительной информации.
        """

        self.current_candidate = self.candidates[self.current_step]

        if self.candidates[self.current_step] >= self.best_candidate_so_far:
            self.best_candidate_so_far = self.current_candidate

        if action == 1:  # предложенный кандидат выбран
            terminated = True  # останавливаем процесс
            reward = 100 if self.ranks[self.current_step] == 0 else -1

        else:  # отвергаем кандидата
            if self.current_step == self.num_candidates - 1:  # если это последний кандидат - выбираем его
                terminated = True
                reward = 100 if self.ranks[self.current_step] == 0 else -1
            else:
                terminated = False
                reward = 0
                #  Прокручиваем список кандидатов
                self.current_step += 1

        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        return observation, reward, terminated, truncated, info
