import re
import numpy as np

import pygraphviz
import networkx as nx
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
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

def colorFader(mix=0, c1="blue", c2="red"):
    # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def parse_file(lines, verbose=0):
    cluster_info = {}

    d = 0
    for i, cluster in enumerate(lines.split("# Cluster:")[1:]):

        if i+1 not in cluster_info:
            cluster_info[i+1] = {}

        name = re.findall(r"\[(.*?)]", cluster)[0]
        if verbose:
            print(name)
        if name == "noise":
            l = 0  # Note: this is later corrected to = d
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
        cluster_info[i + 1]['points'] = np.subtract(IDs, 1)

        cluster_info[i + 1]['parents'] = []

        points = cluster.split("ID=")
        if d == 0 and len(points) > 0:
            d = len(points[1].split()) - 1

    noise_index = get_index(cluster_info, 0, 0)
    cluster_info[noise_index]['lambda'] = d

    for i, cluster in enumerate(lines.split("# Cluster:")[1:]):

        children_list = re.findall(r"# Children: (.*?) ID", cluster)
        if children_list:
            children_list = children_list[0].strip().split(" ")
            cluster_info[i + 1]['parents'] = []

            for child in children_list:
                inds = child.replace("[", "").replace("]", "").split("_")
                child_index = get_index(cluster_info, int(inds[0]), int(inds[1]))
                if child_index == -1:
                    print("****")
                if 'parents' not in cluster_info[child_index]:
                    cluster_info[child_index]['parents'] = []
                cluster_info[child_index]['parents'].append(i+1)

    return cluster_info

def draw_graph(cluster_info, ax_ij):


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

        added += 1
        pos[name] = [cluster_info[i]['lambda'], cluster_info[i]["index"]]
        attr[name] = cluster_info[i]['lambda']
        G.add_node(name, level=cluster_info[i]['lambda'])


        class_1 = np.count_nonzero(np.array(cluster_info[i]['points']) > 499)
        percentage = class_1/len(cluster_info[i]['points'])
        colors.append(colorFader(percentage))

    for i in cluster_info.keys():
        name = str(cluster_info[i]['lambda']) + "_" + str(cluster_info[i]["index"])

        for j in cluster_info[i]['parents']:
            name_parent = str(cluster_info[j]['lambda']) + "_" + str(cluster_info[j]["index"])
            G.add_edge(name_parent, name)

    pos = nx.random_layout(G)
    nx.set_node_attributes(G, attr, name="level")

    true_levels = nx.get_node_attributes(G, 'level')
    reposition = {node_id: np.array([pos[node_id][0], true_levels[node_id]]) for node_id in true_levels}

    n = np.arange(0, d+1)

    # for i in n:
    #     ax_ij.axhline(y=i, color='gray', linewidth=0.5)

    G.remove_node(f"{d}_0")
    noise_index = get_index(cluster_info, d, 0)
    colors.pop(noise_index-1)

    nx.draw_networkx(G, reposition, node_size=60, arrowsize=1, width=0.1, node_color=colors, with_labels=False, ax=ax_ij)

    # plt.show()





