from collections import defaultdict

import numpy as np
import dill as pickle

from environment import BoardEnv


class BotPlayer:
    """Агент обучения с подкреплением - игрок в крестики-нолики"""

    def __init__(
            self,
            env: BoardEnv,
            epsilon_decay: float,
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            final_epsilon: float = 0,
            q_values_file: str = '',
            discount_factor: float = 0.95,
    ):
        """Инициализирует агента обучения с подкреплением алгоритма Q-learning.

        :param env: Среда обучения
        :param learning_rate: Скорость обучения
        :param initial_epsilon: Начальное значение эпсилон
        :param epsilon_decay: Скорость уменьшения эпсилон
        :param final_epsilon: Конечное значение эпсилон
        :param discount_factor: Коэффициент дисконтирования для вычисления Q-значения
        """

        self.env = env

        if q_values_file == '':
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
            self.read_from_file = False
        else:
            self.q_values = self.load_q_table(q_values_file)
            self.read_from_file = True

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple) -> int:
        """Возвращает лучшее действие с вероятностью (1 - epsilon),
        в противном случае — случайное действие с вероятностью эпсилон для обеспечения исследования.

        :param obs: Текущее состояние (наблюдение).
        :return: Действие из пространства действий.
        """

        if self.read_from_file:
            return int(np.argmax(self.q_values[obs]))
        else:
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """Обновляет Q-таблицу после действия (action).

        :param obs: Текущее состояние (наблюдение)
        :param action: Выполненное действие
        :param reward: Полученное вознаграждение
        :param terminated: Флаг завершения эпизода
        :param next_obs: Следующее состояние (наблюдение)
        """

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_q_table(self, filename):
        """Сохранение Q-таблицы в файл

        :param filename: путь к файлу
        :return:
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.q_values, file)

    def load_q_table(self, filename):
        """Чтение Q-таблицы из файла

        :param filename: путь к файлу
        :return:
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)
