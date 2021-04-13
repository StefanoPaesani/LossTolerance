import numpy as np

imaginary_part_cutoff = 10 ** (-14)


## TODO: use numpy/itertools maps instead of for loops

def check_z_phase(n, state, nqbts, offset_phase=0):
    """ A function that, given the vector of a graph state, determines if the n-th qubit has a local phase (Z rotation)
    In case the local phase is np.pi, it corresponds to a self-loop.

    It does so by looking at the element in the vector associated to the ket of the type |00010...00>, where all
    elements are 0 except the n-th, which is 1. The phase of such term(up to the offset, which
    ensures that the term |00..00> has sign +1) corresponds to the local Z phase.
    """
    return np.angle(state[(2 ** (nqbts - n - 1))]) - offset_phase


def check_if_neighbours(n, k, state, local_phases, nqbts, offset_phase=0):
    """ A function that, given the vector state of a graph state, determines whether qubits
    n and k are neighbours in the graph.

    It does so by looking at the element in the vector associated to the ket of the type |00010...010>, where all
    elements are 0 except the k-th and the n-th, which are 1s. The sign of such term is -1 (up to the offset and
    local phases, which ensures that the term |00..00> has sign +1) iff there is a CZ (thus an edge) between qubits n
    and k: given that all the other qubits are 0s, the only way there is -1 is if k and n are the control and target
    of a CZ.
    """
    if np.isclose(
            (np.angle(state[(2 ** (nqbts - k - 1)) + (2 ** (nqbts - n - 1))]) - (
                    local_phases[n] + local_phases[k] + offset_phase)) % (
                    2 * np.pi),
            np.pi):
        return True
    else:
        return False


def find_adj_matrix(state, num_qbts=None):
    """ A function that, if state_signs are the signs of the vector state of a graph state, determinines the adjacency
    matrix of the graph (i.e. determines which graph state it is). Requires num_qbts(num_qbts-1)/2 operations.

    It works by applying the check_if_neighbours function to reconstruct the off-diagonal elements of the adjacency
    matrix.
    """

    ref_phase = np.angle(state[0])

    if num_qbts:
        nqbts = num_qbts
    else:
        nqbts = int(np.log2(len(state)))

    local_phases_list = [check_z_phase(n, state, nqbts, offset_phase=ref_phase) for n in range(nqbts)]
    adj_mat = np.zeros((nqbts, nqbts))
    for n in range(nqbts - 1):
        for k in range(n + 1, nqbts):
            if check_if_neighbours(n, k, state, local_phases_list, nqbts, offset_phase=ref_phase):
                adj_mat[n, k] = adj_mat[k, n] = 1
    return adj_mat, local_phases_list


def check_correct_signs(state, adj_mat, local_phases, num_qbts=None, print_error=False):
    """ A function that, given a vector state and the adjacency matrix of a graph, determines whether the vector
    corresponds to the graph state associated to the graph.

    If the state is a n-qubit state it requires O(2^n) operations, because it needs to check the sign of all elements
    in the state (it does it exactly once). I don't think it can be done with less operations, e.g. because changing the
    sign of a single element in the vector still provides a physical state, but that does not longer correspond to a
    graph.

    It works in the following way. For the coefficient associated to each component |x>=|x1 x2 ... xn> is given by,
    the sign is given by (-1)**(number of pairs of neighbours, i.e. that have an edge between them, such that their
    associated bits in the binary expansion of x are 1). In fact, for each pair with qubits in |1>, an edge corresponds
    to a CZ, which gives a -1 sign. Therefore, to check if the sign of each component is correct, we have to count how
    the pairs of neighbours we have in the qubits corresponding to the 1s in the binary expansion of x.
    On top of this we also check that all the elements have the same amplitude. If both conditions are satisfied for
    all terms in the vector state, then it corresponds to the graph state.

    If print_error is True, it prints what type of error is preventing the state to be a graph state, if any.
    """
    if num_qbts:
        nqbts = num_qbts
    else:
        nqbts = int(np.log2(len(state)))

    ref_phase = np.angle(state[0])
    ref_ampl = np.abs(state[0])

    # print('Starting! will use:')
    # print('ref_phase', ref_phase)
    # print('ref_ampl', ref_ampl)
    # print('local_phases', local_phases)
    # print('adj_mat:')
    # print(adj_mat)

    # elem indicates all comp. basis terms |00..00>, |00..01>,..,|11..11>, with qubit states given by the binary
    # expansion of elem.
    for elem in range(2 ** nqbts):

        # print('\nDoing elem', bin(elem)[::-1])

        # finds positions of 1s in the binary expansion of elem
        pos_ones = [pos for pos, bit in enumerate(bin(elem)[2:].zfill(nqbts)) if bit == '1']
        # print('pos_ones', pos_ones)

        # calculate offset due to local phases
        total_local_phases_offset = sum([local_phases[i] for i in pos_ones]) + ref_phase
        # print('total_local_phases_offset', total_local_phases_offset)

        # counts the number of neighbouring pairs of qubits in this that are 1s in elems. Initialised to 0.
        num_neigh_pairs = 0

        this_ampl = state[elem]

        # print('this_ampl', this_ampl)
        # print('abs(this_ampl)', abs(this_ampl))
        # print('this_angle', np.angle(this_ampl))

        # check if the amplitude gives non-uniform amplitudes.
        if not np.isclose(np.abs(this_ampl), np.abs(ref_ampl)):
            if print_error:
                print('Amplitudes not uniform')
            return False

        for n in range(len(pos_ones)):
            for k in range(n + 1, len(pos_ones)):
                if adj_mat[pos_ones[n], pos_ones[k]] == 1:
                    num_neigh_pairs += 1
        # print('num_neigh_pairs', num_neigh_pairs)

        # checks if the sign corresponds to (-1)^(number of neighbouring pairs with qubits in 1)
        # tries to avoid having to use complex numbers

        if not np.isclose(
                (np.angle(this_ampl) - total_local_phases_offset) % (2 * np.pi),
                (np.pi * num_neigh_pairs) % (2 * np.pi)):
            if print_error:
                print('Found a minus sign in the wrong position')
            return False

    # If all the terms have a correct sign, the state is the graph state corresponding to the input adjacency matrix,
    # and the function returns True.
    return True


def vector_is_graphstate(state, num_qbts=None, print_error=False):
    r""" Function that checks whether a given vector state represents a graph state. If it does, it returns the
    adjacency matrix of the graph.

    It works by first using the find_adj_matrix function, which if state is a graph state obtains the adjacency matrix,
    and then using the check_correct_signs function, which tests whether or not state corresponds to the graph state
    associated to an input adjacency matrix. The output is True iff state is a graph state: if it is a graph state first
    we obtain its adjacency matrix adj_matrix so that check_correct_signs returns True when using adj_matrix, while if
    it is not a graph state, check_correct_signs returns False for any input adj_matrix, and in particular for the
    one returned by find_adj_matrix.

    :param list state: a 2^n long vector representing the n-qubit state
    :param int num_qbts: used to specify the number of qubits, without having to calculate it from the vector.
    :param bool print_error: When True, prints what error is preventing the vector from representing a graph state,
    if any error occurs.
    """
    if num_qbts:
        nqbts = num_qbts
    else:
        nqbts = int(np.log2(len(state)))

    adj_matrix, local_phases = find_adj_matrix(state, num_qbts=nqbts)
    # print()
    # print('adj_matrix')
    # print(adj_matrix)
    # print()
    # print('local_phases')
    # print(local_phases)
    # print()

    test_graphstate = check_correct_signs(state, adj_matrix, local_phases, num_qbts=nqbts, print_error=print_error)

    if test_graphstate:
        return True, adj_matrix, local_phases
    else:
        return False, [], []


if __name__ == '__main__':

    ####################################################
    #######        TEST INDIVIDUAL VECTORS      ########
    ####################################################

    # mystate = np.array([1, 1, 1, -1, 1, 1, -1, 1])  # three qubit line
    # mystate = np.array([1, 1, 1, -1, -1, -1, 1, -1])  # three qubit line - with loop on last qubit
    # mystate = np.array([1, 1, 1, -1, 1, -1, 1, 1])  # three qubit line - ordered as 0-2-1
    # mystate = np.array([1, 1, 1, -1,  1, -1, -1, -1]) # three qubit cycle

    # mystate = np.array([1, 1, 1, -1, 1, -1, -1, -1])
    # mystate = np.array([0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j, # five qubit line
    #           0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #           0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,
    #          -0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j, -0.17677668+0.j,
    #           0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,
    #           0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #          -0.17677668+0.j, -0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #           0.17677668-0.j,  0.17677668-0.j, -0.17677668+0.j,  0.17677668-0.j])

    # three qubit line with z rotation on first qubit
    # global_phase = np.pi/8.
    # theta = np.pi/3.
    # mystate = np.exp(1.j*global_phase) * np.kron(np.array([[1, 0], [0, np.exp(1.j * theta)]]), np.identity(2 ** (2))) @ \
    # np.array([1, 1, 1, -1, 1, 1, -1, 1])

    # three qubit line with z rotation on first qubit
    # global_phase = np.pi
    # theta = 0
    # mystate = np.exp(1.j*global_phase) * np.kron(np.array([[1, 0], [0, np.exp(1.j * theta)]]), np.identity(2 ** (3))) @ \
    #           np.ones(16)

    # three qubit line with z rotation on first and last qubit
    # theta1 = np.pi/4.
    # theta3 = 3*np.pi/4
    # mat1 = np.array([[1, 0], [0, np.exp(1.j * theta1)]])
    # mat3 = np.array([[1, 0], [0, np.exp(1.j * theta3)]])
    # mystate = np.kron(np.kron(mat1, np.identity(2)), mat3) @ np.array([1, 1, 1, -1, 1, 1, -1, 1])

    # five qubit line with z rotation on first qubit
    # theta = np.pi/4.
    # mystate = np.kron(np.array([[1, 0], [0, np.exp(1.j*theta)]]), np.identity(2**(4))) @ \
    #           np.array([0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,
    #           0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #           0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,
    #          -0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j, -0.17677668+0.j,
    #           0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,
    #           0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #          -0.17677668+0.j, -0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #           0.17677668-0.j,  0.17677668-0.j, -0.17677668+0.j,  0.17677668-0.j])

    #### Test state

    # print(mystate)
    #
    # test_result, Amat, local_phases = vector_is_graphstate(mystate, print_error=True)
    #
    # print(test_result)
    # print(Amat)
    # print(local_phases)
    #
    # if test_result:
    #     from CodesFunctions.GraphStateClass import GraphState
    #     import networkx as nx
    #     import matplotlib.pyplot as plt
    #
    #     graph = nx.from_numpy_matrix(Amat)
    #     gstate = GraphState(graph)
    #     gstate.image()
    #     plt.show()

    #######################################################
    ### TESTS WITH ALL POSSIBLE N-qubit GRAPHS VECTORS  ###
    #######################################################

    import networkx as nx
    from CodesFunctions.GraphStateClass import GraphState
    from CodesFunctions.LTCodeClass import powerset
    from itertools import product, combinations
    import matplotlib.pyplot as plt

    qubits_num = 5

    graph_nodes = list(range(qubits_num))
    all_possible_edges = combinations(graph_nodes, 2)
    all_graphs_by_edges = list(powerset(all_possible_edges))
    num_graphs = len(all_graphs_by_edges)

    recognised_graphs_num = 0

    ### add global phase
    global_phase = np.pi / 7

    ### add Z rotation on first qubit
    theta = np.pi / 5
    added_U = np.kron(np.array([[1, 0], [0, np.exp(1.j * theta)]]), np.identity(2 ** (qubits_num - 1)))

    ### print stuff?
    print_tests = False

    for edges_conf_ix, graph_edges in enumerate(all_graphs_by_edges):
        # print('Testing graph', edges_conf_ix, 'of', num_graphs)
        graph = nx.Graph()
        graph.add_nodes_from(graph_nodes)
        graph.add_edges_from(graph_edges)
        this_gstate = GraphState(graph)

        target_adjmat = this_gstate.adj_mat()

        vect_state0 = this_gstate.graph_vecstate()
        vect_state = added_U @ vect_state0
        vect_state = np.exp(1.j * global_phase) * vect_state
        test_if_graph, Amat, _ = vector_is_graphstate(vect_state, print_error=False)

        if print_tests:
            print()
            print('target_adjmat:')
            print(target_adjmat)
            print('vect_state0')
            print(vect_state0)
            print('Amat')
            print(Amat)
            print('vect_state')
            print(vect_state)
            print('test_if_graph')
            print(test_if_graph)

            gstate = GraphState(graph)
            gstate.image()
            plt.show()

        if test_if_graph:
            # if test_if_graph and np.array_equal(Amat, target_adjmat):
            recognised_graphs_num += 1

    print('\n\nSuccessful graphs recognized: ', recognised_graphs_num * 100. / num_graphs, '%')
