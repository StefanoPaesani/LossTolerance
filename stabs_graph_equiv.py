import numpy as np
from linear_algebra_inZ2 import row_echelon_inZ2
from GraphStateClass import GraphState
from StabStateClass import StabState


def stab_to_graph(stab_state):
    r""" Function that finds the graph state associated to a given stabilizer state.
    Based on: Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 69, 022316 (2004)

    :param stab_state: The stabilizer state to be converted into a graph state.
    :type stab_state: :class:`StabState`
    """
    if not isinstance(stab_state, StabState):
        raise ValueError("Input needs to be a StabState class.")

    stab_gens = stab_state.stab_gens

    num_gen = len(stab_gens)
    num_qbts = stab_state.nqbits

    clifford_transf = ['I' for i in range(num_qbts)]

    mat = stab_state.as_binary()
    # print(mat)

    X_mat = mat[num_qbts:]
    # print(X_mat)

    test_mat, inv_mat_Tr = row_echelon_inZ2(X_mat.T)
    inv_mat = inv_mat_Tr.T

    # finds the positions of linearly independent rows in R_x by obtaining the column of the first 1 for each row
    # that start with a 1. Such columns are linearly independent because we have a reduced row echolon form (this is
    # is equivalent to get the rows associated with columns that start with a 1 in R_x)
    lin_ind_rows = [next((i for i, x in enumerate(this_row) if x), None) for this_row in test_mat if np.any(this_row)]
    compl_rows = [x for x in range(num_qbts) if x not in lin_ind_rows]

    mat = (mat @ inv_mat) % 2

    # swap i-th and (i+n)-th rows for each i in compl_rows,
    # corresponding applying an Hadamard to the qubits i for i in compl_rows
    for qbt_idx in compl_rows:
        clifford_transf[qbt_idx] = 'H'
        mat[[qbt_idx, qbt_idx + num_qbts]] = mat[[qbt_idx + num_qbts, qbt_idx]]

    # find inverse of X'
    X_mat1 = mat[num_qbts:]
    Z_mat1 = mat[:num_qbts]
    test_mat1, inv_mat_Tr1 = row_echelon_inZ2(X_mat1.T)
    inv_mat1 = inv_mat_Tr1.T

    # obtains total basis transformation R to be applied on the right of the stabilizer binary matrix
    basis_change_mat = (inv_mat @ inv_mat1) % 2

    # obtains the adjacency_matrix, possibly with 1s in the diagonal
    adj_mat = (Z_mat1 @ inv_mat1) %2

    for qbt_idx in range(num_qbts):
        if adj_mat[qbt_idx, qbt_idx] == 1:
            adj_mat[qbt_idx, qbt_idx] = 0
            clifford_transf[qbt_idx] = 'S'+clifford_transf[qbt_idx]

    G = nx.from_numpy_matrix(adj_mat)
    graph_state = GraphState(G)

    return graph_state, adj_mat, clifford_transf, basis_change_mat


if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    from StabStateClass import StabState
    from GraphStateClass import GraphState
    import qecc as q

    # stab_gens = ["XZII", "ZXZI", "IZXZ", "IIZX"]
    # stab_gens = ["ZZII", "XXZI", "IZXZ", "IIZX"]  ##H on first
    # stab_gens = ["ZXII", "XZZI", "IXXZ", "IIZX"]  ##H on first and second
    # stab_gens = ["ZZXXX", "XXXXX", "XZZXX", "XXZZX", "XXXZZ"] # GHZ
    stab_gens = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ", "ZZZZZ"]  ## 5-qubit code
    # stab_gens = ["ZZIIIIIII", "ZIZIIIIII", "IIIZZIIII", "IIIZIZIII", "IIIIIIZZI", "IIIIIIZIZ",
    #              "XXXXXXIII", "XXXIIIXXX", "ZIIIZIZII"]  ## Shor code

    gen_list = q.PauliList(stab_gens)
    stab_state = StabState(gen_list)

    graph_state, adj_mat, clifford_transf, basis_change_mat = stab_to_graph(stab_state)

    print('Adjacency matrix of equivalent graph state:')
    print(adj_mat)
    print('Local Clifford transformation:')
    print(clifford_transf)

    # SOME TARGET GRAPHS

    ######## 4 qubit line
    # graph_targ = nx.Graph()
    # graph_targ.add_nodes_from([0, 1, 2, 3])
    # graph_targ.add_edges_from([(0, 1), (1, 2), (2, 3)])

    ######## graph for 5-qubit star
    # graph_targ = nx.Graph()
    # graph_targ.add_nodes_from([0, 1, 2, 3, 4])
    # graph_targ.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])

    ######## graph for 5-qubit code
    graph_targ = nx.Graph()
    graph_targ.add_nodes_from([0, 1, 2, 3, 4])
    graph_targ.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

    ######## graph for Shore code, from Griffiths notes
    # graph_targ = nx.Graph()
    # graph_targ.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # graph_targ.add_edges_from([(0, 1), (1, 2), (2, 0), (0,3 ),(0,4), (1,5), (1,6), (2,7),(2,8)])





    # TEST LOCAL EQUIVALENCE WITH TARGET GRAPH

    graph_state_targ = GraphState(graph_targ)
    check_equiv, unitaries = graph_state_targ.is_LC_equiv(graph_state, return_all=True)
    print('Are the two graphs locally equivalent? Which Clifford operators transform them into each other?')
    print(check_equiv)
    print(unitaries)




    ###### PLOT GRAPH EQUIVALENT TO STABILIZER CODE
    plt.subplot(211)
    graph_state.image(with_labels=True)
    plt.subplot(212)
    graph_state_targ.image(with_labels=True)
    plt.show()


