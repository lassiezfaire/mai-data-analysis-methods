import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import SecretaryAgent
from environment import CandidateEnv

num_candidates = 100
env = CandidateEnv(num_candidates=num_candidates)

# Гиперпараметры
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = SecretaryAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

ranks_of_chosen = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # оборот цикла - эпизод
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # обновляем агента
        agent.update(obs, action, reward, terminated, next_obs)

        # обновляем флаг завершения эпизода и текущее состояние
        done = terminated
        obs = next_obs

    ranks_of_chosen.append(info['rank of current'])
    agent.decay_epsilon()

plt.figure(figsize=(10, 6))
plt.hist(ranks_of_chosen, bins=num_candidates)
plt.xticks(np.arange(0, num_candidates + 1, 10))
plt.xlabel('Chosen candidate')
plt.ylabel('frequency')
plt.show()
