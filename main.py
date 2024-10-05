import json
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def get_zero_lvl_ids(filename: str = 'zero_lvl_ids.txt') -> list[str]:
    """

    :param filename: имя файла, содержащего id уровня 0
    :return:
    """

    zero_lvl_ids = []
    with open(filename, "r") as f:
        for line in f:
            zero_lvl_ids.append(line.strip())
    return zero_lvl_ids


def get_first_second_lvl_ids(zero_lvl_ids: list[str], filename: str = 'friends.json'):
    """Проверяет наличие файла с информацией о друзьях. Если такого нет, создаёт его и наполняет информацией

    :param filename: Имя файла с информацией о друзьях. Требуется именно .json-файл установленного образца
    :return: словарь, содержащий информацию из файла
    """

    graph = {}

    if os.path.exists(filename):
        if is_valid_json(filename=filename):
            print(f"Читаем информацию из существующего {filename}...")
            # Читаем существующий файл с друзьями

            with open(filename, 'r') as f:
                friends_graph = json.load(f)
                return friends_graph
        else:
            print(f"Файл {filename} повреждён. Удалите его и перезапустите скрипт")
    else:
        print(f"Создаём новый {filename}...")
        # Получаем моих друзей

        print('Всего найдено id уровня 0', len(zero_lvl_ids) - 1)
        for zero_lvl_id in zero_lvl_ids:
            first_and_second_lvl_ids = get_friends(zero_lvl_id, access_token)
            graph[zero_lvl_id] = [str(user_id) for user_id in first_and_second_lvl_ids.get('items')]

            friend_list = first_and_second_lvl_ids.get('items')

            print("Всего найдено друзей:", first_and_second_lvl_ids.get('count'))

            # Получаем друзей друзей
            for user_id in tqdm(friend_list):
                user_friends = get_friends(user_id, access_token)
                try:
                    graph[str(user_id)] = [str(user_id) for user_id in user_friends.get('items')]
                except:
                    graph[str(user_id)] = ['']
                time.sleep(1)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=4)

        return graph


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
        print(f"Error: {data['error']['error_msg']}")
        return []

    return data['response']


def is_valid_json(filename: str = 'friends.json'):
    """Проверяет, является ли файл настоящим JSON.

    :param filename: Имя файла, который нужно проверить.
    :return: True, если файл является корректным JSON-файлом, иначе False.
    """

    try:
        with open(filename, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False


access_token = os.getenv('TOKEN')
zero_lvl_ids = get_zero_lvl_ids()

first_second_lvl_ids = get_first_second_lvl_ids(zero_lvl_ids=zero_lvl_ids)

# friends_graph = work_with_file()
#
# # Создаем направленный граф
# G = nx.from_dict_of_lists(friends_graph)
#
# # Оцениваем граф по центральности
# betweenness_centrality = nx.betweenness_centrality(G)  # по посредничеству
# closeness_centrality = nx.closeness_centrality(G)  # по близости
# eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=600)  # собственного вектора
#
# # Вывод результатов
# print("Центральность по посредничеству:", {key: round(value, 2) for key, value in betweenness_centrality.items()})
# print("Центральность по близости:", {key: round(value, 2) for key, value in closeness_centrality.items()})
# print("Центральность собственного вектора:", {key: round(value, 2) for key, value in eigenvector_centrality.items()})
#
# print(type(betweenness_centrality))
#
# # Чертим граф
# print("Чертим граф...")
#
# plt.figure(figsize=(80, 80))
# pos = nx.spring_layout(G, k=0.1)  # k можно менять для лучшего эффекта
# # pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
#
# nx.draw(G,
#         pos,
#         with_labels=True,
#         node_size=20,
#         node_color='lightblue',
#         font_size=5,
#         font_weight='normal',
#         arrows=False,
#         width=0.5)
#
# plt.title("Граф друзей ВКонтакте")
# plt.show()
