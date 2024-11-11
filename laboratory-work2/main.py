import matplotlib.pyplot as plt

from math_solution.main import math_solution
from reinforcement_learning.main import reinforcement_learning

num_candidates = 100

# print('Прогресс математического алгоритма:')
math_solution_results = math_solution(num_candidates=num_candidates)

# print('Прогресс алгоритма обучения с подкреплением:')
reinforcement_learning_results = reinforcement_learning(num_candidates=num_candidates)

bins = list(range(0, num_candidates + 1, 10))

plt.figure(figsize=(10, 6))

plt.hist(math_solution_results, bins=num_candidates, alpha=0.5, label='reinforcement learning')
plt.hist(reinforcement_learning_results, bins=num_candidates, alpha=0.5, label='reinforcement learning')
plt.xlabel('Chosen candidate')
plt.ylabel('frequency')
plt.show()
