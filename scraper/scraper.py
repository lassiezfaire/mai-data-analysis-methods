import json
import os
import time

from tqdm import tqdm
import requests


def get_zero_lvl_ids(filepath: str) -> list[str]:
    """
    Загружает информацию о профилях, чьих друзей мы будем анализировать

    :param filepath: путь к файлу, содержащему id уровня 0
    :return: list id уровня 0
    """

    zero_lvl_ids = []

    try:
        with open(filepath, "r") as f:
            for line in f:
                zero_lvl_ids.append(line.strip())

        print('Всего найдено id уровня 0:', len(zero_lvl_ids))
        return zero_lvl_ids
    except FileNotFoundError:
        print(f"Не найден файл анализируемых профилей {filepath}")


def get_first_second_lvl_ids(zero_lvl_ids: list[str],
                             access_token: str,
                             filepath: str):
    """Проверяет наличие файла с информацией о друзьях. Если такого нет, создаёт его и наполняет информацией

    :param zero_lvl_ids: Список id группы
    :param access_token: Токен доступа, получается указанным в документации образом
    :param filepath: Имя файла с информацией о друзьях. Требуется именно .json-файл установленного образца
    :return: словарь, содержащий информацию из файла
    """

    graph = {}
    zero_lvl_ids = [zero_lvl_id for zero_lvl_id in zero_lvl_ids if zero_lvl_id != '']

    if os.path.exists(filepath):
        if is_valid_json(filepath=filepath):
            print(f"Файл {filepath} содержит id уровня 1 и 2. Считываем информацию...")
            # Читаем существующий файл с друзьями

            with open(filepath, 'r') as f:
                friends_graph = json.load(f)
                return friends_graph
        else:
            print(f"Файл {filepath} повреждён. Удалите его и перезапустите скрипт")
    else:
        print("Файл, содержащий id уровня 1 и 2, не найден.")
        print(f"Информация об id уровня 1 и 2 будет сохранена в файл {filepath}.")

        # Получаем друзей
        for zero_lvl_id in zero_lvl_ids:
            try:  # На случай невозможности получить список друзей (приватный либо удалённый профиль)
                first_and_second_lvl_ids = get_friends(zero_lvl_id, access_token)
                graph[zero_lvl_id] = [str(user_id) for user_id in first_and_second_lvl_ids.get('items')]

                friend_list = first_and_second_lvl_ids.get('items')

                print(
                    f"Всего найдено id уровня 1 для пользователя {zero_lvl_id}: "
                    f"{first_and_second_lvl_ids.get('count')}")
                time.sleep(1)

                # Получаем друзей друзей
                progress_bar = tqdm(friend_list)
                for user_id in progress_bar:
                    user_friends = get_friends(user_id, access_token)
                    try:
                        graph[str(user_id)] = [str(user_id) for user_id in user_friends.get('items')]
                    except AttributeError:
                        graph[str(user_id)] = ['']
                    progress_bar.set_description(f"Поиск id уровня 2 для пользователя {user_id}")
                    time.sleep(0.1)

            except AttributeError:
                graph[zero_lvl_id] = ['']

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=4)

        return graph


def is_valid_json(filepath: str):
    """Проверяет, является ли файл настоящим JSON.

    :param filepath: Имя файла, который нужно проверить.
    :return: True, если файл является корректным JSON-файлом, иначе False.
    """

    try:
        with open(filepath, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False


def get_friends(user_id: str, access_token: str):
    """Возвращает список друзей пользователя ВКонтакте.

    :param user_id: ID пользователя ВКонтакте.
    :param access_token: Токен доступа пользователя.
    :return: Список словарей, где каждый словарь содержит id друзей пользователя
    """

    url = 'https://api.vk.com/method/friends.get'
    params = {
        'user_id': user_id,
        'access_token': access_token,
        'v': '5.199'
    }

    response = requests.post(url, params=params)
    data = response.json()

    if 'error' in data:
        return ['']

    return data['response']
