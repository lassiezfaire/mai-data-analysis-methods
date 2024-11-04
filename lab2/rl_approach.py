import gymnasium as gym
import numpy as np


class SecretaryProblemEnv(gym.Env):
    def __init__(self, n: int):
        super().__init__()
        self.n = n  # Количество кандидатов
        self.observation_space = gym.spaces.Discrete(n + 1)
        self.action_space = gym.spaces.Discrete(2)
        self.reset()

    def reset(self) -> int:
        self.current_candidate = 0
        self.best_candidate_so_far = None
        self.candidates = np.arange(1, self.n + 1)
        np.random.shuffle(self.candidates)  # Генерируем рандомные значения для кандидатов
        self.done = False
        return self.current_candidate

    def step(self, action):
        reward = 0
        if action == 1:  # Принять кандидата
            if self.best_candidate_so_far is None:
                self.best_candidate_so_far = self.candidates[self.current_candidate]
            else:
                if self.candidates[self.current_candidate] > self.best_candidate_so_far:
                    self.best_candidate_so_far = self.candidates[self.current_candidate]
            self.done = True
        self.current_candidate += 1
        if self.current_candidate == self.n:
            self.done = True
        if self.done:
            reward = self.best_candidate_so_far
        observation = self.current_candidate if not self.done else self.n
        print("")
        return observation, reward, self.done, {}

    def render(self, mode='human'):
        print(f"Текущий кандидат: {self.current_candidate}")
        if self.best_candidate_so_far is not None:
            print(f"Лучший кандидат: {self.best_candidate_so_far}")


def train_secretary_model_qlearning(env, num_episodes, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
    """
    Обучение модели для решения задачи о секретаре с использованием Q-learning.

    Args:
        env: Объект окружения для задачи о секретаре.
        num_episodes: Количество эпизодов обучения.
        learning_rate: Скорость обучения (alpha).
        discount_factor: Фактор дисконтирования (gamma).
        epsilon: Вероятность случайного выбора действия (epsilon-жадность).
    """

    # Инициализация Q-таблицы
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    rewards = []
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Выбор действия (epsilon-жадность)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Случайное действие
            else:
                action = np.argmax(q_table[observation])  # Наилучшее действие

            # Выполнение шага в окружении
            next_observation, reward, done, info = env.step(action)
            total_reward += reward

            # Обновление Q-таблицы
            q_table[observation, action] = (1 - learning_rate) * q_table[observation, action] + \
                                           learning_rate * (
                                                       reward + discount_factor * np.max(q_table[next_observation]))

            observation = next_observation

        rewards.append(total_reward)

    # Вывод средней награды
    average_reward = np.mean(rewards)
    print(f"Средняя награда за {num_episodes} эпизодов: {average_reward:.4f}")


# Пример использования:
env = SecretaryProblemEnv(100)
train_secretary_model_qlearning(env, num_episodes=1000)
