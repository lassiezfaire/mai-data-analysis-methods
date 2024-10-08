import os

from dotenv import load_dotenv
import networkx as nx

from scraper import scraper
from centrality import centrality
from graph_draw import graph_draw

load_dotenv()

access_token = os.getenv('TOKEN')

print('### START OF TRANSMISSION ###\n')

zero_lvl_ids = scraper.get_zero_lvl_ids(filepath='data/zero_lvl_ids.txt')  # получаем id уровня 0
# получаем id уровня 1 для расчёта центральностей
# first_lvl_ids = scraper.get_first_second_lvl_ids(zero_lvl_ids=zero_lvl_ids,
#                                                  access_token=access_token,
#                                                  filepath='data/first_second_lvl_ids.json',
#                                                  second_id_lvl=False)
# получаем id уровня 1 и 2
first_second_lvl_ids = scraper.get_first_second_lvl_ids(zero_lvl_ids=zero_lvl_ids,
                                                        access_token=access_token,
                                                        filepath='data/full_first_second_lvl_ids.json',
                                                        second_id_lvl=True)

print("")

# G = nx.from_dict_of_lists(first_lvl_ids)  # Создаём граф друзей и друзей друзей
full_G = nx.from_dict_of_lists(first_second_lvl_ids)  # Создаём граф друзей и друзей друзей

centrality_dict = centrality.centrality(graph=full_G, filepath='data/centrality.json')  # Считаем центральности
# Выводим центральности для членов банды
centrality.band_centrality(band_members_number=2,
                           zero_lvl_ids=zero_lvl_ids,
                           centrality_dict=centrality_dict)

print("")

node_sizes = graph_draw.node_sizes(graph=full_G, centrality=centrality_dict, filepath='data/node_sizes.txt')
graph_draw.graph_draw(graph=full_G, node_sizes=node_sizes, filepath="test.html")

print('\n### END OF TRANSMISSION ###')
