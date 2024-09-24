import os
import time

from dotenv import load_dotenv
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

load_dotenv()


# Функция API по получению друзей
def get_friends(user_id, access_token):
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


friends_graph = {}

my_id = '711398942'
access_token = os.getenv('TOKEN')

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

# Создаем направленный граф
G = nx.DiGraph()

# Добавляем рёбра в граф
for key, values in friends_graph.items():
    for value in values:
        G.add_edge(key, value)

print("Чертим граф...")
# Рисуем граф
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Позиции вершин
nx.draw(G,
        pos,
        with_labels=False,
        node_size=20,
        node_color='lightblue',
        font_size=10,
        font_weight='bold',
        arrows=False,
        width=0.5)
plt.title("Граф друзей ВКонтакте")
plt.show()

