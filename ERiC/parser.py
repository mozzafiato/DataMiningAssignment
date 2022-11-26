import re
import numpy as np

import pygraphviz
import networkx as nx
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import random

def read_file(file):
    with open(file) as f:
        lines = " ".join(line.strip() for line in f)
    return lines


def get_index(cluster_info, l, c_i):
    for i in cluster_info.keys():
        if cluster_info[i]['lambda'] == l and cluster_info[i]['index'] == c_i:
            return i
    return -1


def parse_file(lines):
    cluster_info = {}

    for i, cluster in enumerate(lines.split("# Cluster:")[1:]):
        #print("Cluster", i+1)
        if i+1 not in cluster_info:
            cluster_info[i+1] = {}

        name = re.findall(r"\[(.*?)]", cluster)[0]
        print(name)
        if name == "noise":
            # not sure ??
            l = 0
            c_i = 0

            cluster_info[i + 1]['lambda'] = int(l)
            cluster_info[i + 1]['index'] = int(c_i)
        else:
            split = name.split("_")
            l = split[0]
            c_i = split[1]

        cluster_info[i + 1]['lambda'] = int(l)
        cluster_info[i + 1]['index'] = int(c_i)

        IDs_list = re.findall(r"ID=(\d+)", cluster)
        IDs = np.squeeze(np.array(IDs_list, dtype="i").reshape(-1, 1))
        cluster_info[i + 1]['points'] = IDs

    for i, cluster in enumerate(lines.split("# Cluster:")[1:]):

        children_list = re.findall(r"# Children: (.*?) ID", cluster)
        if children_list:
            children_list = children_list[0].strip().split(" ")
            cluster_info[i + 1]['parents'] = []

            for child in children_list:
                inds = child.replace("[", "").replace("]", "").split("_")
                #print(inds)
                child_index = get_index(cluster_info, int(inds[0]), int(inds[1]))
                if child_index == -1:
                    print("****")
                if 'parents' not in cluster_info[child_index]:
                    cluster_info[child_index]['parents'] = []
                #print("Adding parent:", i+1, " to", child_index)
                cluster_info[child_index]['parents'].append(i+1)

    return cluster_info


def draw_graph(cluster_info):

    G = nx.DiGraph()
    pos = {}
    attr = {}
    colors = []
    d = max([cluster_info[c]['lambda'] for c in cluster_info])

    # add nodes:
    print("Keys:", len(list(cluster_info.keys())))
    added = 0
    for i in cluster_info.keys():

        name = str(cluster_info[i]['lambda']) + "_" + str(cluster_info[i]["index"])
        G.add_node(name, level=cluster_info[i]['lambda'])
        added += 1
        print(added)
        pos[name] = [cluster_info[i]['lambda'], cluster_info[i]["index"]]
        print("Node:", name, " level:", cluster_info[i]['lambda'])
        if cluster_info[i]['lambda'] == 0:
            attr[name] = 0
        else:
            attr[name] = d - cluster_info[i]['lambda']+1

        if cluster_info[i]['lambda'] == 0:
            colors.append("black")
        else:
            class_1 = np.count_nonzero(cluster_info[i]['points'] > 500)
            class_2 = len(cluster_info[i]['points']) - class_1
            if class_1 > class_2:
                colors.append('red')
            else: colors.append('blue')

    for i in cluster_info.keys():
        name = str(cluster_info[i]['lambda']) + "_" + str(cluster_info[i]["index"])

        #print("Node:", name)
        for j in cluster_info[i]['parents']:
            name_parent = str(cluster_info[j]['lambda']) + "_" + str(cluster_info[j]["index"])
            G.add_edge(name_parent, name)
            #print("Adding:", name_parent, name)

    """
    # test
    G.add_node("2_1", level=2)
    G.add_edge("1_0", "2_1")
    attr["2_1"] = 2
    G.add_node("2_2", level=2)
    G.add_edge("1_0", "2_2")
    attr["2_2"] = 2
    G.add_node("3_1", level=3)
    G.add_edge("1_0", "3_1")
    attr["3_1"] = 1
    G.add_edge("2_1", "3_1")
    """

    lambda_max = max([cluster_info[c]['lambda'] for c in cluster_info])
    #cluster_info[noise_index]['lambda'] = lambda_max

    pos = graphviz_layout(G)
    nx.set_node_attributes(G, attr, name="level")

    true_levels = nx.get_node_attributes(G, 'level')
    reposition = {node_id: np.array([pos[node_id][0], true_levels[node_id]]) for node_id in true_levels}
    print(G.number_of_nodes())
    print(G.number_of_edges())
    print(added)

    n = np.arange(0, lambda_max+1)

    for i in n:
        plt.axhline(y=i, color='gray', linewidth=0.5)

    nx.draw_networkx(G, reposition, node_size=60, arrowsize=1, width=0.05, node_color=colors, with_labels=False)

    plt.show()

# Example
text = read_file('elki/elki_output.txt')
cluster_info = parse_file(text)
draw_graph(cluster_info)





