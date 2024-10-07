import json
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def get_zero_lvl_ids(filename: str = 'zero_lvl_ids.txt') -> list[str]:
    """
    Загружает информацию о профилях, чьих друзей мы будем анализировать

    :param filename: имя файла, содержащего id уровня 0
    :return:
    """

    zero_lvl_ids = []
    with open(filename, "r") as f:
        for line in f:
            zero_lvl_ids.append(line.strip())
    return zero_lvl_ids


def get_first_second_lvl_ids(zero_lvl_ids: list[str], filename: str = 'first_second_lvl_ids.json'):
    """Проверяет наличие файла с информацией о друзьях. Если такого нет, создаёт его и наполняет информацией

    :param filename: Имя файла с информацией о друзьях. Требуется именно .json-файл установленного образца
    :param zero_lvl_ids: Список id группы
    :return: словарь, содержащий информацию из файла
    """

    graph = {}
    zero_lvl_ids = [zero_lvl_id for zero_lvl_id in zero_lvl_ids if zero_lvl_id != '']

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

        print('Всего найдено id уровня 0:', len(zero_lvl_ids) - 1)
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
                    except:
                        graph[str(user_id)] = ['']
                    progress_bar.set_description(f"Поиск id уровня 2 для пользователя {user_id}")
                    time.sleep(0.1)
            except:
                graph[zero_lvl_id] = ['']

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

    error = 0

    if 'error' in data:
        # print(f"Error: {data['error']['error_msg']}")
        error += 1
        return ['']

    return data['response']


def is_valid_json(filename: str = 'first_second_lvl_ids.json'):
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


def centralities_to_file(graph: nx.Graph, filename: str = 'centralities.json'):
    """

    :param graph: Граф, центральности узлов которого будут сохранены в файл
    :param filename: Пусть к файлу, куда будет сохранена центральность графа
    """

    if os.path.exists(filename):
        if is_valid_json(filename=filename):
            print(f"Читаем информацию из существующего {filename}...")

            # Читаем существующий файл центральностей
            with open(filename, 'r') as f:
                centralities = json.load(f)
                return centralities
        else:
            print(f"Файл {filename} повреждён. Удалите его и перезапустите скрипт.")
    else:
        print(f"Считаем центральности графа друзей, вершин: {graph.number_of_nodes()}, "
              f"рёбер: {graph.number_of_edges()}")
        centralities = {}

        ig_graph = ig.Graph.from_networkx(graph)  # Конвертируем в формат более быстрой библиотеки

        progress_bar = tqdm(ig_graph.vs)
        for node in progress_bar:
            centralities[node["_nx_name"]] = []

            betwenness = round(ig_graph.betweenness(vertices=[node.index], directed=False, cutoff=3)[0], 2)
            centralities[node["_nx_name"]].append(betwenness)
            closeness = round(ig_graph.closeness(vertices=[node.index])[0], 3)
            centralities[node["_nx_name"]].append(closeness)
            eigenvector = round(nx.eigenvector_centrality(graph, max_iter=600)[node["_nx_name"]], 3)
            centralities[node["_nx_name"]].append(eigenvector)

            progress_bar.set_description(f'Центральность для id {node["_nx_name"]}')

        with open(filename, 'w') as f:
            json.dump(centralities, f, indent=4)

        return centralities


access_token = os.getenv('TOKEN')
zero_lvl_ids = get_zero_lvl_ids()

first_second_lvl_ids = get_first_second_lvl_ids(zero_lvl_ids=zero_lvl_ids)

# Создаем граф
G = nx.from_dict_of_lists(first_second_lvl_ids)

# Считаем центральности
centralities = centralities_to_file(graph=G)

number_of_band_members = 1  # количество людей в команде, для которых будет оцениваться центральность
band_centrality = {
    'id': [],
    'betweenness': [],
    'closeness': [],
    'eigenvector': []
}  # центральности в банде

for band_memeber in zero_lvl_ids[0:number_of_band_members]:
    band_centrality['id'].append(band_memeber)
    band_centrality['betweenness'].append(centralities[band_memeber][0])
    band_centrality['closeness'].append(centralities[band_memeber][1])
    band_centrality['eigenvector'].append(centralities[band_memeber][2])

df = pd.DataFrame(band_centrality)
print(df.to_string(index=False))

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
