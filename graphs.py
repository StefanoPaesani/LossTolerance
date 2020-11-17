import networkx as nx
# TODO: use graph-tool instead of networkx...much faster for graphs (and better graphics), but only for Mac/Linux

import numpy as np
from itertools import product, combinations


## GRAPH INITIALIZATION FUNCTIONS ##

def gen_linear_graph(nqubits):
    graph = nx.Graph()
    graph.add_nodes_from(range(nqubits))
    these_edges = [(node_ix, node_ix + 1) for node_ix in range(nqubits - 1)]
    graph.add_edges_from(these_edges)
    return graph


def gen_ring_graph(nqubits):
    graph = nx.Graph()
    graph.add_nodes_from(range(nqubits))
    these_edges = [(node_ix, (node_ix + 1) % nqubits) for node_ix in range(nqubits)]
    graph.add_edges_from(these_edges)
    return graph


def gen_star_graph(nqubits, central_qubit=0):
    graph = nx.Graph()
    nodes = range(nqubits)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        product([central_qubit], [other_nodes for other_nodes in nodes if other_nodes != central_qubit]))
    return graph


def gen_fullyconnected_graph(nqubits):
    graph = nx.Graph()
    nodes = range(nqubits)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(combinations(nodes, 2))
    return graph


def gen_crazy_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix*nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        these_edges = product(nodes_mat[layer_ix], nodes_mat[layer_ix + 1])
        graph.add_edges_from(these_edges)
    return graph



def gen_multiwire_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix*nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        these_edges = zip(nodes_mat[layer_ix], nodes_mat[layer_ix + 1])
        graph.add_edges_from(these_edges)
    return graph
