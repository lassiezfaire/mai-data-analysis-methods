import numpy as np
import matplotlib.pyplot as plt


def classic_algorythm(n: int = 100):
    """

    :param n: Количество женихов
    :return: Найденный лучший жених
    """

    candidates = np.arange(1, n + 1)
    np.random.shuffle(candidates)

    stop = int(round(n / np.e))
    best_from_rejected = np.min(candidates[:stop])
    rest = candidates[stop:]

    try:
        return rest[rest < best_from_rejected][0]
    except IndexError:
        return candidates[-1]


sim = np.array([classic_algorythm() for i in range(100000)])

plt.figure(figsize=(10, 6))
plt.hist(sim, bins=100)
plt.xticks(np.arange(0, 101, 10))
plt.ylim(0, 40000)
plt.xlabel('Chosen candidate')
plt.ylabel('frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(np.histogram(sim, bins=100)[0]) / 100000)
plt.ylim(0, 1)
plt.xlim(0, 100)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(0, 101, 10))
plt.xlabel('Chosen candidate')
plt.ylabel('Cumulative probability')
plt.show()
