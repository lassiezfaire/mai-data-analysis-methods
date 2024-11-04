import os
import math

import networkx as nx
from bokeh.models import Circle
from bokeh.plotting import figure, from_networkx, output_file, show


def graph_draw(graph: nx.Graph, node_sizes: list[float], filepath: str):
    """

    :param graph: Граф, который будет отрисован
    :param node_sizes: размеры вершины, рассчитанные специальной функцией
    :param filepath: Указывает, куда сохранить файл
    """
    print("Чертим граф друзей. Это небыстрый процесс. Наберитесь терпения.")

    # Создание фигуры Bokeh
    plot = figure(title="Граф друзей ВКонтакте",
                  tools="pan,wheel_zoom,reset",
                  width=1000,
                  height=1000)

    graph = from_networkx(graph, nx.spring_layout(graph, k=0.09), scale=3, center=(0, 0))

    graph.node_renderer.data_source.data['node_sizes'] = node_sizes
    graph.node_renderer.glyph = Circle(radius='node_sizes')

    plot.renderers.append(graph)

    # Отображение графика
    show(plot)

    # Сохранение изображения
    output_file(filepath)


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
            node_values[key] = value[1]

        sizes = []
        for node in graph.nodes:
            try:
                # sizes.append(node_values[node] / 50)
                if node_values[node] / 50 >= 0.01:
                    sizes.append(node_values[node] / 35)
                elif node_values[node] / 50 < 0.0071:
                    sizes.append(node_values[node] / 500)
                else:
                    sizes.append(node_values[node] / 50)
            except KeyError:
                sizes.append(0.005)

        with open(filepath, 'w', encoding='utf-8') as file:
            for item in sizes:
                file.write(str(item) + '\n')

        return sizes
