import os
import time

from dotenv import load_dotenv
import requests

load_dotenv()

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


user_id = '711398942'
access_token = os.getenv("TOKEN")

friends = get_friends(user_id, access_token)
print(f"Друзья пользователя {user_id}: {friends}")


friend_list = friends.get('items')

for user_id in friend_list:
    friends = get_friends(user_id, access_token)
    print(f"Друзья пользователя {user_id}: {friends}")
    time.sleep(1)

