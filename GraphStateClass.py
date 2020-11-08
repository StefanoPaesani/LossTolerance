## IMPORTS ##

# Used only in benchmarking
import time

import numpy as np
from itertools import product, combinations
import networkx as nx
# TODO: use graph-tool instead of networkx...much faster for graphs (and better graphics), but requires Mac/Linux

import cirq
# See comments in GraphState.graph_vecstate to see why/if cirq is required

import qecc as q

import matplotlib.pyplot as plt

## import graph-state specific functions
from lc_equivalence import check_LCequiv


## CLASSES ##

class GraphState(object):
    r"""
    Class representing a Graph state on :math:`n` qubits.

    :param graph: The graph representing the state, where each node represents a qubit and edges are entangling gates
    :type graph: :class:`nx.Graph`
    """

    def __init__(self, graph):
        # graph is an object from the Graph class of networkx

        # Check that the graph is the correct object
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input graph needs to be a Graph object of NetworkX.")

        self.graph = graph

        # calculates the stabilizer generators of the graph
        self.stab_gens = stabilizer_generators_from_graph(graph)

    def __hash__(self):
        # We need a hash function to store GraphStates as dict keys or in sets.
        return hash(self.graph)

    def __len__(self):
        """
        Yields the number of qubits in the graph.
        """
        return len(self.graph.nodes())

    ## PRINTING ##
    def image(self, with_labels=True, font_weight='bold'):
        """
        Produces a matplotlib image of the graph associated to the graph state.
        """
        return nx.draw(self.graph, with_labels=with_labels, font_weight=font_weight)

    ## REPRESENTATION ##

    def graph_vecstate(self):
        """
        Calculates the vector state associate to the graph, and returns it as a numpy array.
        """

        # TODO: use faster way to calculate state vectors for stabilizer states, that does not pass through circuit
        #  simulation (cirq). See e.g. Aaronson & Gottensman arXiv:quant-ph/0406196. Still, using cirq for
        #  calculating the state of a graph is much faster than calculating it from the stabilized state
        #  [qecc.PauliList.stabilizer_subspace()] (0.01s vs 8s for 15 qubits crazygraph on my laptop, 21 qubits
        #  crazygraph takes 0.5s with cirq). Alternatively there is a  function of the PauliList
        #  in the Granade package (stabilizer_subspace) which does it, but it is really too slow (takes 10 seconds
        #  already with 7 qubits).

        graph_nodes = list(self.graph.nodes())
        nqubits = len(graph_nodes)

        qubits = cirq.LineQubit.range(nqubits)

        CZlist = [cirq.CZ(qubits[graph_nodes.index(edge_ix[0])], qubits[graph_nodes.index(edge_ix[1])])
                  for edge_ix in self.graph.edges()]
        circuit = cirq.Circuit(cirq.H.on_each(*qubits), CZlist)

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        return result.final_state_vector

    ## Equivalence under Local Complementation ##

    def is_LC_equiv(self, other, return_all=True):
        r""" Function that checks whether the graph state is locally equivalent to another one,
        If they are, it provides the local operations to be performed to convert one into the other.
        The algorithm runs in O(V^4) (V: #vertices).
        Based on: Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 70, 034302 (2004)

        :param other: The second graph.
        :type other: :class:`GraphState`
        :param bool return_all: When True, the algorithm returns all possible Clifford unitaries that provide a
        LC equivalence between the two graphs. When False it only returns the first it finds.
        """
        return check_LCequiv(self.graph, other.graph, return_all=return_all)


## STABILIZER FUNCTIONS ##

def stabilizer_generators_from_graph(graph):
    r"""
    Calculates the stabilizer generators associated to a graph .

    :param graph: The graph representing the state, where each node represents a qubit and edges are entangling gates
    :type graph: :class:`nx.Graph`
    :returns: A :class:`qecc.PauliList` object representing the set of :math:`[K_1,...,K_n]` stabilizer generators
    """
    stab_gens = []
    nodes = list(graph.nodes())
    nqubits = len(nodes)
    for node_ix in nodes:
        stab_dict = {nodes.index(node_ix): 'X'}
        for ngb_node_ix in graph.neighbors(node_ix):
            stab_dict[nodes.index(ngb_node_ix)] = 'Z'
        this_stab = q.Pauli.from_sparse(stab_dict, nq=nqubits)
        stab_gens.append(this_stab)
    return q.PauliList(stab_gens)


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
    graph.add_nodes_from(range(nrows * nlayers))
    for layer_ix in range(nlayers - 1):
        these_edges = product(nodes_mat[layer_ix], nodes_mat[layer_ix + 1])
        graph.add_edges_from(these_edges)
    return graph


if __name__ == '__main__':
    ########### DEFINE GRAPHS TO USE

    #### Linear vs fully connected graphs - 3 qubits
    # nqb = 3
    # G = gen_linear_graph(nqb)
    # G_2 = gen_fullyconnected_graph(nqb)

    #### Linear vs fully connected graphs - 4 qubits
    # nqb = 4
    # G = gen_linear_graph(nqb)
    # G_2 = gen_fullyconnected_graph(nqb)

    #### Star vs fully connected graphs - 4 qubits
    # nqb = 4
    # G = gen_star_graph(nqb)
    # G_2 = gen_fullyconnected_graph(nqb)

    #### Star vs Star with relabeling - 4 qubits
    # nqb = 4
    # G = gen_star_graph(nqb)
    # G_2 = gen_star_graph(nqb, central_qubit=1)

    #### Linear vs standard ring - 4 qubits
    # nqb = 4
    # G = gen_linear_graph(nqb)
    # G_2 = gen_ring_graph(nqb)

    #### Linear vs standard ring + relabeling - 4 qubits
    # nqb = 4
    # G = gen_linear_graph(nqb)
    # temp_G_2 = gen_ring_graph(nqb)
    # G_2 = nx.relabel_nodes(temp_G_2, {1: 2, 2: 1})

    #### Star vs Fully connected, large graphs - n qubits
    nqb = 22
    G = gen_fullyconnected_graph(nqb)
    G_2 = gen_star_graph(nqb)

    gstate = GraphState(G)
    gstate_2 = GraphState(G_2)

    ### Obtain the stabilizer generators of the graph G
    print('Stabilizer generators of G:')
    print(gstate.stab_gens)
    print(gstate_2.stab_gens)

    ### Checks the amount of time to calculate the state vector of the graph state (and prints it, if uncommented)
    start = time.time()
    state_from_cirq = gstate.graph_vecstate()
    end = time.time()
    print('Time required to calculate the state vector:', end - start, 's')
    ## print(np.around(state_from_cirq, 3))

    ### Checks if the two graph states are equivalent under local complementation,
    ### and which local operations are required
    start = time.time()
    check_equiv, unitaries = gstate.is_LC_equiv(gstate_2, return_all=True)
    end = time.time()
    print('Are the two graphs locally equivalent? Which Clifford operators transform them into each other?')
    print(check_equiv)
    print(unitaries)
    print('Time required to check LC equivalence:', end - start, 's')

    ### Print the graphs associated to the states
    plt.subplot(211)
    gstate.image(with_labels=True)
    plt.subplot(212)
    gstate_2.image(with_labels=True)
    plt.show()

################ OTHER TESTS

    # nqb = 5
    # G = gen_linear_graph(nqb)
    # # G = gen_ring_graph(nqb)
    #
    # gstate = GraphState(G)
    # print(gstate.graph_vecstate())
    # print(gstate.stab_gens)
    #
    # gstate.image(with_labels=True)
    # plt.show()
