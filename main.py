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
# получаем id уровня 1 и 2
first_second_lvl_ids = scraper.get_first_second_lvl_ids(zero_lvl_ids=zero_lvl_ids,
                                                        access_token=access_token,
                                                        filepath='data/first_second_lvl_ids.json')

print("")

G = nx.from_dict_of_lists(first_second_lvl_ids)  # Создаём граф друзей
centrality_dict = centrality.centrality(graph=G, filepath='data/centrality.json')  # Считаем центральности
# Выводим центральности для членов банды
centrality.band_centrality(band_members_number=1,
                           zero_lvl_ids=zero_lvl_ids,
                           centrality_dict=centrality_dict)

print("")

node_sizes = graph_draw.node_sizes(graph=G, centrality=centrality_dict, filepath='data/node_sizes.txt')
graph_draw.graph_draw(graph=G, node_sizes=node_sizes)

print('\n### END OF TRANSMISSION ###')
