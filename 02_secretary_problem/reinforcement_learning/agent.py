from collections import defaultdict
from typing import Any, Dict
import numpy as np

from reinforcement_learning.environment import CandidateEnv


class SecretaryAgent:
    """Агент обучения с подкреплением - сущность, которая будет решать задачу и обучаться в процессе"""

    def __init__(
            self,
            env: CandidateEnv,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Инициализирует агента обучения с подкреплением с пустым словарем значений состояния-действия (q_values),
        скоростью обучения и значением epsilon.

        :param env: Среда обучения
        :param learning_rate: Скорость обучения
        :param initial_epsilon: Начальное значение эпсилон
        :param epsilon_decay: Скорость уменьшения эпсилон
        :param final_epsilon: Конечное значение эпсилон
        :param discount_factor: Коэффициент дисконтирования для вычисления Q-значения
        """

        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: Dict[str, Any]) -> int:
        """Возвращает лучшее действие с вероятностью (1 - epsilon),
        в противном случае — случайное действие с вероятностью эпсилон для обеспечения исследования.

        :param obs: Текущее состояние (наблюдение).
        :return: Действие из пространства действий.
        """

        hashible_obs = tuple(obs.values())

        if np.random.random() < self.epsilon:
            return obs['if candidate is best so far']
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
        """Обновляет Q-таблицу после действия (action).

        :param obs: Текущее состояние (наблюдение)
        :param action: Выполненное действие
        :param reward: Полученное вознаграждение
        :param terminated: Флаг завершения эпизода
        :param next_obs: Следующее состояние (наблюдение)
        """

        hashible_obs = tuple(obs.values())
        hashable_next_obs = tuple(next_obs.values())

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
