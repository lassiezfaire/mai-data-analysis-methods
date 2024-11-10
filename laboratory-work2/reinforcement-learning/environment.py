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
    """Среда обучения - мир
    (кандидаты в количестве num_candidates штук),
    в котором будет действовать агент."""

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
            'if candidate is best so far': self.if_best_so_far,
            'share of rejected': round(self.current_step / self.num_candidates * 100)
        }

    def _get_info(self) -> dict:
        """Диагностические данные, которые позволяют отслеживать работу нейронной сети

        :return: словарь диагностической информации
        """
        return {
            'rank of current': find_key_by_value(
                dictionary=self.candidates,
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

        self.candidates = generate_candidate_dict(num_candidates=self.num_candidates)  # список кандидатов
        self.best_candidate = self.candidates[1]  # лучший кандидат (имеющий ранг 1)
        print(self.candidates[1])

        self.current_step = 0  # порядковый номер текущего кандидата в списке кандидатов, обновляется в step()
        self.current_candidate = list(self.candidates.values())[self.current_step]  # текущий кандидат

        self.if_best_so_far = False  # проверка, является ли текущий кандидат лучшим
        self.best_candidate_so_far = 0  # лучший из отвергнутых кандидатов (без ранга), обновляется в step()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action) -> Tuple[dict, int, bool, bool, dict]:
        """Выполняет действие, вычисляет новое состояние среды и возвращает кортеж из следующего наблюдения,
        вознаграждения, признаков завершения и дополнительной информации.

        :param action: Действие, выполняемое агентом.
        :return: Кортеж из следующего наблюдения, вознаграждения, признаков завершения и дополнительной информации.
        """

        self.current_candidate = list(self.candidates.values())[self.current_step]

        if self.best_candidate_so_far < self.current_candidate:
            self.best_candidate_so_far = self.current_candidate
            self.if_best_so_far = True

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
                reward = 0
                # добавляем пропущенного кандидата в список пропущенных кандидатов
                self.current_step += 1

        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        return observation, reward, terminated, truncated, info
