import os

import matplotlib.pyplot as plt
import networkx as nx


def graph_draw(graph: nx.Graph, node_sizes: list[float]):
    """

    :param graph: Граф, который будет отрисован
    :param node_sizes: размеры вершины, рассчитанные специальной функцией
    """
    print("Чертим граф друзей. Это небыстрый процесс. Наберитесь терпения.")

    plt.figure(figsize=(80, 80))
    pos = nx.spring_layout(graph, k=0.09)

    nx.draw(graph,
            pos,
            with_labels=False,
            node_size=node_sizes,
            node_color='lightblue',
            font_size=5,
            font_weight='normal',
            arrows=False,
            width=0.5)

    plt.title("Граф друзей ВКонтакте")
    plt.show()


def node_sizes(graph: nx.Graph, centrality: dict, filepath: str) -> list[float]:
    """

    :param graph: Граф, для которого обрабатываются центральности
    :param centrality: Словарь, содержащий центральности
    :param filepath: Имя файла, содержащего размеры точек
    :return:
    """
    if os.path.exists(filepath):
        print(f"Файл {filepath} содержит размеры вершин графа. Считываем информацию...")

        sizes = []
        with open(filepath, "r") as f:
            for line in f:
                sizes.append(float(line.strip()))

        return sizes
    else:
        print(f"Файл, содержащий размеры вершин графа, не найден.")
        print(f"Информация о размерах вершин графа будет сохранена в файл {filepath}.")

        node_values = {}
        for key, value in centrality.items():
            node_values[key] = value[0]

        sizes = []
        for node in graph.nodes:
            try:
                sizes.append(node_values[node] / 1000)
            except KeyError:
                sizes.append(1000)

        with open(filepath, 'w', encoding='utf-8') as file:
            for item in sizes:
                file.write(str(item) + '\n')

        return sizes
