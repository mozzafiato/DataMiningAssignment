import numpy as np    
from itertools import chain

def ncut_single_graph(graph, nodes1, nodes2):
    """
    Computes the NCut-value for a binary partition of a networkx Graph.
    
    graph: a networkx graph instance with edge weights
    nodes1: list of nodes in the first partition
    nodes2: list of nodes in the second partition
    
    output: the normalized cut size
    """
    
    # find edges between the two partitions
    edges = [edge for edge in graph.edges if
                    (edge[0] in nodes1 and edge[1] in nodes2) or
                    (edge[1] in nodes1 and edge[0] in nodes2)]

    weights = [graph.get_edge_data(edge[0], edge[1])['weight'] for edge in edges]
    
    #print(weights)
    cut_size = sum(weights)
    
    # find edges connected to the first partition 
    edges = [edge for edge in graph.edges if
                    edge[0] in nodes1] + [edge for edge in graph.edges if
                    edge[1] in nodes1]

    weights = [graph.get_edge_data(edge[0], edge[1])['weight'] for edge in edges]
    assoc1 = sum(weights)
       
    # find edges connected to the second partition 
    edges = [edge for edge in graph.edges if
                    edge[0] in nodes2] + [edge for edge in graph.edges if
                    edge[1] in nodes2]
    weights = [graph.get_edge_data(edge[0], edge[1])['weight'] for edge in edges]
    assoc2 = sum(weights)
    
    return cut_size/assoc1 + cut_size/assoc2
    
def ncut_multigraph(graph, nodes1, nodes2):
    """
    Computes the NCut-value for a binary partition of a networkx Multigraph.
    Assumes that all edge weights are 1.
    
    graph: a networkx multigraph instance
    nodes1: list of nodes in the first partition
    nodes2: list of nodes in the second partition
    
    output: the normalized cut size
    """
    
    # find edges between the two partitions
    edges = [edge for edge in graph.edges if
                    (edge[0] in nodes1 and edge[1] in nodes2) or
                    (edge[1] in nodes1 and edge[0] in nodes2)]

    cut_size = len(edges) # assumes that all edges have a weight of 1
    
    # find edges connected to the first partition 
    edges = [edge for edge in graph.edges if
                    edge[0] in nodes1] + [edge for edge in graph.edges if
                    edge[1] in nodes1]
    assoc1 = len(edges) # assumes that all edges have a weight of 1
       
    # find edges connected to the second partition 
    edges = [edge for edge in graph.edges if
                    edge[0] in nodes2] + [edge for edge in graph.edges if
                    edge[1] in nodes2]
    assoc2 = len(edges) # assumes that all edges have a weight of 1
    
    return cut_size/assoc1 + cut_size/assoc2
    
def k_ncut_multigraph(graph, node_lists):
    """
    Computes the NCut-value for a k-way partition of a networkx Multigraph.
    The algorithm is based on the NCut_k-value from https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf.
    Assumes that all edge weights are 1.
    
    graph: a networkx multigraph instance
    node_lists: a numpy array created from a list of k lists of nodes that define the partitions
    
    output: the normalized cut size
    """
    
    k_ncut = 0
    
    for i, nodes in enumerate(node_lists):
        other_nodes = np.concatenate(np.concatenate((node_lists[:i], node_lists[i+1:])))
        
        # find edges between the current subset and all other nodes
        edges = [edge for edge in graph.edges if
                        (edge[0] in nodes and edge[1] in other_nodes) or
                        (edge[1] in nodes and edge[0] in other_nodes)]
                        
        cut_i = len(edges) # assumes that all edges have a weight of 1
        
        # # find edges connected to the current subset 
        edges = [edge for edge in graph.edges if
                        edge[0] in nodes] + [edge for edge in graph.edges if
                        edge[1] in nodes]
        assoc_i = len(edges) # assumes that all edges have a weight of 1
        
        
        if assoc_i != 0:
            cut_i /= assoc_i
            
        k_ncut += cut_i
        
    return k_ncut