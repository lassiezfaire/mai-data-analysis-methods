import json
import os

from tqdm import tqdm
import igraph as ig
import networkx as nx
import pandas as pd

from scraper import scraper


def centrality(graph: nx.Graph, filepath: str):
    """

    :param graph: Граф, центральности узлов которого будут сохранены в файл
    :param filepath: Пусть к файлу, куда будет сохранена центральность графа
    :return: Словарь центральностей графа
    """

    if os.path.exists(filepath):
        if scraper.is_valid_json(filepath=filepath):
            print(f"Файл {filepath} содержит центральности вершин графа. Считываем информацию...")

            # Читаем существующий файл центральностей
            with open(filepath, 'r') as f:
                centrality_dict = json.load(f)
                return centrality_dict
        else:
            print(f"Файл {filepath} повреждён. Удалите его и перезапустите скрипт.")
    else:
        print("Файл, содержащий центральности вершин графа, не найден.")
        print(f"Считаем центральности вершин графа друзей, вершин: {graph.number_of_nodes()}, "
              f"рёбер: {graph.number_of_edges()}.")
        print(f"Информация будет сохранена в файл {filepath}.")
        centrality_dict = {}

        ig_graph = ig.Graph.from_networkx(graph)  # Конвертируем в формат более быстрой библиотеки

        progress_bar = tqdm(ig_graph.vs)
        for node in progress_bar:
            centrality_dict[node["_nx_name"]] = []

            betweenness = round(ig_graph.betweenness(vertices=[node.index], directed=False, cutoff=3)[0], 2)
            centrality_dict[node["_nx_name"]].append(betweenness)
            closeness = round(ig_graph.closeness(vertices=[node.index])[0], 3)
            centrality_dict[node["_nx_name"]].append(closeness)
            eigenvector = round(nx.eigenvector_centrality(graph, max_iter=600)[node["_nx_name"]], 3)
            centrality_dict[node["_nx_name"]].append(eigenvector)

            progress_bar.set_description(f'Центральность для id {node["_nx_name"]}')

        with open(filepath, 'w') as f:
            json.dump(centrality_dict, f, indent=4)

        return centrality_dict


def band_centrality(band_members_number: int, zero_lvl_ids: list[str], centrality_dict: dict):
    """

    :param band_members_number: Количество людей, для которых будет выведена центральность
    :param zero_lvl_ids: Список людей
    :param centrality_dict: Словарь центральностей
    :return:
    """
    band_centrality_dict = {
        'id': [],
        'betweenness': [],
        'closeness': [],
        'eigenvector': []
    }

    for band_member in zero_lvl_ids[0:band_members_number]:
        band_centrality_dict['id'].append(band_member)
        band_centrality_dict['betweenness'].append(centrality_dict[band_member][0])
        band_centrality_dict['closeness'].append(centrality_dict[band_member][1])
        band_centrality_dict['eigenvector'].append(centrality_dict[band_member][2])

    df = pd.DataFrame(band_centrality_dict)
    print(df.to_string(index=False))
