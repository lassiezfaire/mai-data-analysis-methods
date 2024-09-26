import json
import os
import time

from dotenv import load_dotenv
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

load_dotenv()


def get_friends(user_id, access_token):
    """Возвращает список друзей пользователя ВКонтакте.

    Args:
        user_id (str): ID пользователя ВКонтакте.
        access_token (str): Токен доступа пользователя.

    Returns:
        dict: Список словарей, где каждый словарь содержит информацию о друге
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


def is_valid_json(filename):
    """Проверяет, является ли файл настоящим JSON.

    Args:
      filename: Имя файла, который нужно проверить.

    Returns:
      True, если файл является корректным JSON-файлом, иначе False.
    """
    try:
        with open(filename, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False


friends_graph = {}

my_id = '711398942'
access_token = os.getenv('TOKEN')

if os.path.exists('friends.json'):
    if is_valid_json(filename='friends.json'):
        print("Читаем информацию из существующего friends.json...")
        # Читаем существующий файл с друзьями

        with open('friends.json', 'r') as f:
            friends_graph = json.load(f)
    else:
        print("Файл friends.json. Удалите его и перезапустите скрипт")
else:
    print("Создаём новый friends.json...")
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

    with open('friends.json', 'w', encoding='utf-8') as f:
        json.dump(friends_graph, f, indent=4)

# Создаем направленный граф
G = nx.from_dict_of_lists(friends_graph)

print("Чертим граф...")
# Рисуем граф
plt.figure(figsize=(80, 80))
pos = nx.spring_layout(G, k=0.1)  # Позиции вершин
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
