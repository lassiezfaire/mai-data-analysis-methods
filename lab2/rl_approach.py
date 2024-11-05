import gymnasium as gym
import numpy as np


class SecretaryProblemEnv(gym.Env):
    def __init__(self, num_candidates: int = 100):
        super().__init__()
        self.num_candidates = num_candidates
        self.observation_space = gym.spaces.Discrete(num_candidates)
        self.action_space = gym.spaces.Discrete(2)
        self.reset()

    def reset(self, seed=None, options=None):
        self.candidates = array_generator(length=self.num_candidates)
        self.best_candidate_overall = np.max(self.candidates)
        self.current_candidate = 0
        self.best_candidate_so_far = None
        self.done = False
        return self.current_candidate

    def step(self, action):
        reward = 0
        if action == 1:  # Принять кандидата
            self.best_candidate_so_far = self.candidates[self.current_candidate]
            if self.best_candidate_so_far == self.best_candidate_overall:
                reward = 100
            else:
                reward = 1
            self.done = True
        else:  # Отвергнуть кандидата
            if self.current_candidate == self.num_candidates:
                self.best_candidate_so_far = self.candidates[self.current_candidate]
                if self.best_candidate_so_far == self.best_candidate_overall:
                    reward = 100
                else:
                    reward = 1
            self.current_candidate += 1
        observation = self.current_candidate if not self.done else self.num_candidates
        return observation, reward, self.done, {}
    #
    # def render(self, mode='human'):
    #     print(f"Текущий кандидат: {self.current_candidate}")
    #     if self.best_candidate_so_far is not None:
    #         print(f"Лучший кандидат: {self.best_candidate_so_far}")


def array_generator(length):
    random_array = np.random.randint(10 ** 3, (10 ** 4) - 1, size=length)
    unique_array = np.unique(random_array)
    while len(unique_array) < length:
        new_random_numbers = np.random.randint(10 ** 3, (10 ** 4) - 1, size=length - len(unique_array))
        unique_array = np.unique(np.concatenate((unique_array, new_random_numbers)))

    np.random.shuffle(unique_array)

    return unique_array


# def train_secretary_model_qlearning(env, num_episodes, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
#     """
#     Обучение модели для решения задачи о секретаре с использованием Q-learning.
#
#     Args:
#         env: Объект окружения для задачи о секретаре.
#         num_episodes: Количество эпизодов обучения.
#         learning_rate: Скорость обучения (alpha).
#         discount_factor: Фактор дисконтирования (gamma).
#         epsilon: Вероятность случайного выбора действия (epsilon-жадность).
#     """
#
#     # Инициализация Q-таблицы
#     q_table = np.zeros((env.observation_space.n, env.action_space.n))
#     print(q_table.size)
#
#     rewards = []
#     for episode in range(num_episodes):
#         observation = env.reset()
#         done = False
#         total_reward = 0
#
#         while not done:
#             # Выбор действия (epsilon-жадность)
#             if np.random.rand() < epsilon:
#                 action = env.action_space.sample()  # Случайное действие
#             else:
#                 action = np.argmax(q_table[observation])  # Наилучшее действие
#
#             # Выполнение шага в окружении
#             next_observation, reward, done, info = env.step(action)
#             total_reward += reward
#
#             # Обновление Q-таблицы
#             q_table[observation, action] = (1 - learning_rate) * q_table[observation, action] + \
#                                            learning_rate * (
#                                                    reward + discount_factor * np.max(q_table[next_observation - 1]))
#
#             observation = next_observation
#
#         rewards.append(total_reward)
#
#     # Вывод средней награды
#     average_reward = np.mean(rewards)
#     print(rewards)
#     print(f"Средняя награда за {num_episodes} эпизодов: {average_reward:.4f}")


# Пример использования:
env = SecretaryProblemEnv(100)
# train_secretary_model_qlearning(env, num_episodes=1000)
print(f"Количество кандидатов: {len(env.candidates)}")
print(f"Кандидаты:\n {env.candidates}")
print(f"Лучший из кандидатов: {env.best_candidate_overall}")
