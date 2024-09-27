import json
import os
import time

from dotenv import load_dotenv
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx

load_dotenv()


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


def work_with_file(filename: str = 'friends.json'):
    """Проверяет наличие файла с информацией о друзьях. Если такого нет, создаёт его и наполняет информацией

    :param filename: Имя файла с информацией о друзьях. Требуется именно .json-файл установленного образца
    :return: словарь, содержащий информацию из файла
    """

    friends_graph = {}

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

        my_friends = get_friends(my_id, access_token)
        friends_graph[my_id] = [str(user_id) for user_id in my_friends.get('items')]

        friend_list = my_friends.get('items')

        print("Всего найдено друзей:", my_friends.get('count'))

        # Получаем друзей друзей
        for user_id in tqdm(friend_list):
            user_friends = get_friends(user_id, access_token)
            friends_graph[str(user_id)] = [str(user_id) for user_id in user_friends.get('items')]
            time.sleep(1)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(friends_graph, f, indent=4)

        return friends_graph


my_id = os.getenv('MY_ID')
access_token = os.getenv('TOKEN')

friends_graph = work_with_file()

# Создаем направленный граф
G = nx.from_dict_of_lists(friends_graph)
print("Чертим граф...")

# Рисуем граф
plt.figure(figsize=(80, 80))
pos = nx.spring_layout(G, k=0.1)  # k можно менять для лучшего эффекта
# pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

nx.draw(G,
        pos,
        with_labels=True,
        node_size=20,
        node_color='lightblue',
        font_size=5,
        font_weight='normal',
        arrows=False,
        width=0.5)

plt.title("Граф друзей ВКонтакте")
plt.show()
