import networkx as nx

# TODO: delete matplotlib from here
import matplotlib.pyplot as plt


def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def check_isomorphism(G, list_graphs):
    return next((True for elem in list_graphs if nx.is_isomorphic(elem, G)), False)


def is_isomorphic_fixednode(G1, G2, fixed_node):
    nodes = list(G1.nodes)
    nodes2 = list(G2.nodes)
    fixed_edges = G1.edges(fixed_node)
    fixed_edges2 = G2.edges(fixed_node)
    if (nodes != nodes2) or (len(fixed_edges) != len(fixed_edges2)):
        return False
    if fixed_node not in nodes:
        raise ValueError('fixed_node needs to be one of the existing nodes in both graphs')
    nodes.remove(fixed_node)

    ## To check if the two graphs are isomorph taking into account the input qubit, it performs two tests:
    ##  - The graphs are isomorphic
    ##  - The subgraphs without the input node are isomorphic
    is_isomorph_withfixed = nx.is_isomorphic(G1, G2) \
                            and nx.is_isomorphic(G1.subgraph(nodes), G2.subgraph(nodes))


    return is_isomorph_withfixed


def check_isomorphism_with_fixednode(G, list_graphs, fixed_node):
    return next((True for elem in list_graphs if is_isomorphic_fixednode(elem, G, fixed_node)), False)


def local_complementation(G, target):
    """ Function that performs local complementation on a node in a graph"""
    A = nx.Graph(G)
    A_target = nx.ego_graph(A, target, 1, center=False)
    A.remove_edges_from(list(A_target.edges()))
    A_target_comp = nx.complement(A_target)
    A = nx.compose(A, A_target_comp)
    H = nx.Graph(A)
    return H


def lc_equivalence_class_full(G):
    """ Returns the full equivalence class for a graph G, not taking into account any graph isomorphism"""
    lc_class = [G]
    adj_mat_list = [nx.adjacency_matrix(G).todense()]  #
    # for the moment it's using adjacency matrices to check equalities between graphs.
    # Maybe a more compact representation is possible to speed up things?
    nodes = G.nodes
    num_nodes = len(nodes)
    streak_num = 0
    this_lc_node = 0
    while streak_num < num_nodes:
        this_lc_node = this_lc_node % num_nodes
        for this_graph in lc_class:
            new_graph = local_complementation(this_graph, this_lc_node)
            new_adjMat = nx.adjacency_matrix(new_graph).todense()
            if not arreq_in_list(new_adjMat, adj_mat_list):
                lc_class.append(new_graph)
                adj_mat_list.append(new_adjMat)
                streak_num = 0
        this_lc_node += 1
        streak_num += 1
    return lc_class


def lc_equivalence_class(G, fixed_node=None):
    """ Returns the equivalence class for a graph G, where isomorphic graphs are considered to be the equal.
    It also allows to consider a fixed_node which is excluded in the isomorphism check, corresponding to a fixed input
    (e.g. QD spin) qubit for a LTcode.
    """
    lc_class = [G]
    nodes = G.nodes
    num_nodes = len(nodes)
    streak_num = 0
    this_lc_node = 0

    while streak_num < num_nodes:
        this_lc_node = this_lc_node % num_nodes
        for this_graph in lc_class:
            new_graph = local_complementation(this_graph, this_lc_node)
            if fixed_node is None:
                if not check_isomorphism(new_graph, lc_class):
                    lc_class.append(new_graph)
                    streak_num = 0
            else:
                if not check_isomorphism_with_fixednode(new_graph, lc_class, fixed_node):
                    lc_class.append(new_graph)
                    streak_num = 0
        this_lc_node += 1
        streak_num += 1

    return lc_class


if __name__ == '__main__':
    from CodesFunctions.graphs import *
    import matplotlib.pyplot as plt

    ################# TEST GRAPH ISOMORPHISM #####################

    # mygraph0 = gen_star_graph(3)
    # # mygraph0 = gen_linear_graph(5)
    # # mygraph0 = gen_star_graph(5, 0)
    #
    # mygraph1 = gen_ring_graph(3)
    # # mygraph1 = gen_star_graph(5, 2)
    # # mygraph1 = gen_tree_graph([2, 1])
    #
    #
    # are_is = nx.is_isomorphic(mygraph0, mygraph1)
    # if are_is:
    #     print('are isomorphic')
    # else:
    #     print('NOT isomorphic')
    #
    # plt.subplot(1, 2, 1)
    # nx.draw(mygraph0, with_labels=True)
    # plt.subplot(1, 2, 2)
    # nx.draw(mygraph1, with_labels=True)
    # plt.show()

    ################# TEST INDIVIDUAL LOCAL COMPLEMENTATION #####################
    # mygraph0 = gen_tree_graph([2, 2])
    # mygraph0 = gen_star_graph(5)

    # plt.subplot(1, 2, 1)
    # nx.draw(mygraph0, with_labels=True)
    #
    # LCnode = 0
    # mygraphLC = local_complementation(mygraph0, LCnode)
    # plt.subplot(1, 2, 2)
    # nx.draw(mygraphLC, with_labels=True)
    # plt.show()

    ################# FIND FULL LOCAL EQUIVALENCE CLASS #################

    # mygraph0 = gen_star_graph(5)
    # mygraph0 = gen_tree_graph([2, 2])
    # mygraph0 = gen_square_lattice_graph(3, 2)
    # mygraph0 = gen_linear_graph(4)
    mygraph0 = gen_ring_graph(5)

    print('Calculating LC class')
    # lc_class = lc_equivalence_class_full(mygraph0)
    # lc_class = lc_equivalence_class(mygraph0)
    lc_class = lc_equivalence_class(mygraph0, fixed_node=0)

    num_graphs = len(lc_class)
    print('Found', num_graphs, 'graphs in the equivalence class')

    # plot all best graphs
    print('Plotting all graphs in LC class')

    n = num_graphs
    i = 2
    while i * i < n:
        while n % i == 0:
            n = n / i
        i = i + 1

    n_plot_rows = num_graphs / n
    n_plot_cols = n

    for graph_ix, G in enumerate(lc_class):
        plt.subplot(n_plot_rows, n_plot_cols, graph_ix + 1)
        nx.draw(G, with_labels=True)
    plt.show()
