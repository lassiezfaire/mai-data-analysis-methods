import random


def candidate_dictionary(num_candidates: int, seed: int = None) -> dict:
    """Генерируем список кандидатов, потом перемешиваем его

    :param num_candidates: Количество кандидатов
    :param seed:
    :return: словарь кандидатов вида "ранг": "качество"
    """

    random.seed(seed)

    # создаём список кандидатов
    random_candidates = set()
    # генерируем уникальных кандидатов до тех пор, пока их не наберётся num_candidates
    while len(random_candidates) < num_candidates:
        random_candidates.add(random.randint(10 ** 3, 10 ** 4 - 1))

    random_candidates = list(random_candidates)
    # сортируем кандидатов от лучшего к худшему, чтобы рассчитать ранги
    random_candidates.sort(reverse=True)
    # генерируем их ранги
    candidates_ranks = list(range(1, num_candidates + 1))
    # объединяем кандидатов и ранги в словарь кандидатов
    candidates = {}
    for i in range(num_candidates):
        candidates[candidates_ranks[i]] = random_candidates[i]

    # перемешаем кандидатов в случайном порядке с сохранением рангов
    random.shuffle(candidates_ranks)
    shuffled_candidates = {}
    for i in range(num_candidates):
        shuffled_candidates[candidates_ranks[i]] = candidates[candidates_ranks[i]]

    return shuffled_candidates
