import os
import time

from dotenv import load_dotenv
import requests
import matplotlib.pyplot as plt
import networkx as nx

load_dotenv()


#
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

my_friends = get_friends(my_id, access_token)
friends_graph[my_id] = [str(user_id) for user_id in my_friends.get('items')]

friend_list = my_friends.get('items')

for index, user_id in enumerate(friend_list):
    user_friends = get_friends(user_id, access_token)
    friends_graph[str(user_id)] = [str(user_id) for user_id in user_friends.get('items')]
    time.sleep(1)

print(friends_graph)

# Создаем направленный граф
G = nx.DiGraph()

# Добавляем рёбра в граф
for key, values in friends_graph.items():
    for value in values:
        G.add_edge(key, value)

# Рисуем граф
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Позиции вершин
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
plt.title("Граф на основе словаря")
plt.show()

