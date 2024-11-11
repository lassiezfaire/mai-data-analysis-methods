import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from candidates_generator import generate_candidate_dict


def find_key_by_value(dictionary: dict, value):
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    try:
        return keys[values.index(value)]
    except ValueError:
        return -1


def math_solution(num_candidates: int, n_episodes: int = 100_000):
    """

    :param num_candidates: Количество кандидатов
    :param n_episodes: Количество эпизодов
    :return: Список рангов кандидатов, которых программа считает лучшими
    """

    ranks_of_chosen = []

    for episode in tqdm(range(n_episodes)):
        candidates_dict = generate_candidate_dict(num_candidates=num_candidates)
        candidates = list(candidates_dict.values())

        stop = int(round(num_candidates / 2.71828))
        best_from_rejected = max(candidates[:stop])
        rest = candidates[stop:]

        chosen_candidate = 0

        for candidate in rest:
            if candidate > best_from_rejected:
                chosen_candidate = candidate
                break

        if chosen_candidate == 0:
            chosen_candidate = candidates[-1]

        chosen_rank = find_key_by_value(
            dictionary=candidates_dict,
            value=chosen_candidate
        )

        ranks_of_chosen.append(chosen_rank)

    return ranks_of_chosen
