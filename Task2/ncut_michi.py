import numpy as np

def compute_D(graph):
    """
    graph: a networkx graph instance with edge weights
    output: D, a diagonal matrix containing the sum of outgoing edge weights from each node
    """
    D = np.zeros(len(graph.nodes)) # start with a flat array
    
    for i, node in enumerate(graph.nodes):
        outgoing = [edge for edge in graph.edges if edge[0] == node]
        weights = [graph.get_edge_data(edge[0], edge[1])['weight'] for edge in outgoing]
        D[i] = sum(weights)
        
    return np.diag(D)

def compute_W(graph):
    """
    graph: a networkx graph instance with edge weights
    output: W, a symmetric square matrix containing the edge weights
    """
    W = np.zeros((len(graph.nodes), len(graph.nodes)))
    for edge in graph.edges:
        W[edge[0]][edge[1]] = graph.get_edge_data(edge[0], edge[1])['weight']
        W[edge[1]][edge[0]] = graph.get_edge_data(edge[0], edge[1])['weight']
        
    return W

def cut_graph_by_sign(nodes, eigenvec):
    """
    Use the eigenvector with the second smallest eigenvalue to bipartition the graph (e.g. grouping according to sign).
    (see https://en.wikipedia.org/wiki/Segmentation-based_object_categorization) 
    
    nodes: numpy array of nodes in the graph
    eigenvec: numpy array of the eigenvector with the second smallest eigenvalue
    
    output: two numpy arrays of nodes representing graph partitions
    """
    
    nodes1 = nodes[eigenvec >= 0]
    nodes2 = nodes[eigenvec < 0]
    return nodes1, nodes2
    

def ncut_value(graph, nodes1, nodes2):
    """
    The normalized cut size is the cut size times the sum of the reciprocal sizes of the volumes of the two sets.
    
    graph: a networkx graph instance with edge weights
    nodes1: list of nodes in the first partition
    nodes2: list of nodes in the second partition
    
    output: the normalized cut size
    """
    
    # find edges between the two partitions
    edges = [edge for edge in graph.edges if
                    edge[0] in nodes1 and edge[1] in nodes2 or edge or
                    edge[1] in nodes1 and edge[0] in nodes2]
    weights = [graph.get_edge_data(edge[0], edge[1])['weight'] for edge in edges]
    cut_size = sum(weights)
    
    # find edges connected to the first partition 
    edges = [edge for edge in graph.edges if
                    edge[0] in nodes1 or
                    edge[1] in nodes1]
    weights = [graph.get_edge_data(edge[0], edge[1])['weight'] for edge in edges]
    assoc1 = sum(weights)
    
    # find edges connected to the second partition 
    edges = [edge for edge in graph.edges if
                    edge[0] in nodes2 or
                    edge[1] in nodes2]
    weights = [graph.get_edge_data(edge[0], edge[1])['weight'] for edge in edges]
    assoc2 = sum(weights)
    
    return cut_size/assoc1 + cut_size/assoc2