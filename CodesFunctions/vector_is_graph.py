import numpy as np


imaginary_part_cutoff = 10 ** (-14)

## TODO: Make this more efficient, and include other states in the graph basis, using methods from the Griffiths notes.


def check_self_loops(n, state, offset_phase=1, check_no_cmplx=True):
    """ A function that, given the vector of a graph state, determines if the n-th qubit has a self-loop (Z operations)

    It does so by looking at the element in the vector associated to the ket of the type |00010...00>, where all
    elements are 0 except the n-th, which is 1. The sign of such term is -1 (up to the offset, which
    ensures that the term |00..00> has sign +1) iff there is a self-loop on qubit n, because all
    the other qubits are 0s so no CZs are triggered.
    """

    if check_no_cmplx:
        if np.sign(state[(2 ** n)]) == (-1) * offset_phase:
            return 1
        else:
            return 0
    else:  # things get a little bit nastier if complex global phases are involved
        if np.abs(np.exp(np.angle(state[(2 ** n)]) * 1.j) - (-1) * offset_phase) < imaginary_part_cutoff:
            return 1
        else:
            return 0


def check_if_neighbours(n, k, state, self_loops, offset_phase=1, check_no_cmplx=True):
    """ A function that, given the vector state of a graph state, determines whether qubits
    n and k are neighbours in the graph.

    It does so by looking at the element in the vector associated to the ket of the type |00010...010>, where all
    elements are 0 except the k-th and the n-th, which are 1s. The sign of such term is -1 (up to the offset, which
    ensures that the term |00..00> has sign +1) iff there is a CZ (thus an edge) between qubits n and k: given that all
    the other qubits are 0s, the only way there is -1 is if k and n are the control and target of a CZ.
    """

    if check_no_cmplx:
        if np.sign(state[(2 ** k) + (2 ** n)]) == (-1) * offset_phase * ((-1)**(self_loops[n]+self_loops[k])):
            return True
        else:
            return False
    else:  # things get a little bit nastier if complex global phases are involved
        if np.abs(np.exp(np.angle(state[(2 ** k) + (2 ** n)]) * 1.j)*((-1)**(self_loops[n]+self_loops[k])) - (-1) * offset_phase) < imaginary_part_cutoff:
            return True
        else:
            return False


def find_adj_matrix(state, num_qbts=None):
    """ A function that, if state_signs are the signs of the vector state of a graph state, determinines the adjacency
    matrix of the graph (i.e. determines which graph state it is). Requires num_qbts(num_qbts-1)/2 operations.

    It works by applying the check_if_neighbours function to reconstruct the off-diagonal elements of the adjacency
    matrix.
    """
    # tries to avoid using complex numbers
    check_no_cmplx = True
    if isinstance(state[0], np.complex128):
        check_no_cmplx = False
        ref_phase = np.exp(np.angle(state[0]) * 1.j)
    else:
        ref_phase = np.sign(state[0])

    if num_qbts:
        nqbts = num_qbts
    else:
        nqbts = int(np.log2(len(state)))

    self_loop_list = [check_self_loops(n, state, offset_phase=ref_phase, check_no_cmplx=check_no_cmplx) for n in range(nqbts)]
    # adj_mat = np.zeros((nqbts, nqbts))
    adj_mat = np.diag(self_loop_list)
    for n in range(nqbts - 1):
        for k in range(n + 1, nqbts):
            if check_if_neighbours(n, k, state, self_loop_list, offset_phase=ref_phase, check_no_cmplx=check_no_cmplx):
                adj_mat[n, k] = adj_mat[k, n] = 1
    return adj_mat


def check_correct_signs(state, adj_mat, num_qbts=None, print_error=False):
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

    # tries to avoid using complex numbers
    check_no_cmplx = True
    if isinstance(state[0], np.complex128):
        check_no_cmplx = False
        ref_phase = np.exp(np.angle(state[0]) * 1.j)
    else:
        ref_phase = np.sign(state[0])

    ref_ampl = np.abs(state[0])

    self_loops_list = np.diag(adj_mat)

    # elem indicates all comp. basis terms |00..00>, |00..01>,..,|11..11>, with qubit states given by the binary
    # expansion of elem.
    for elem in range(2 ** nqbts):

        # finds positions of 1s in the binary expansion of elem
        pos_ones = [pos for pos, bit in enumerate(bin(elem)[::-1]) if bit == '1']

        # calculate offset due to self-loops
        self_loops_offset = (-1)**sum([self_loops_list[i] for i in pos_ones])

        # counts the number of neighbouring pairs of qubits in this that are 1s in elems. Initialised to 0.
        num_neigh_pairs = 0

        this_ampl = state[elem]

        # check if the amplitude gives non-uniform amplitudes.
        if np.abs(this_ampl) != ref_ampl:
            if print_error:
                print('Amplitudes not uniform')
            return False

        for n in range(len(pos_ones)):
            for k in range(n + 1, len(pos_ones)):
                if adj_mat[pos_ones[n], pos_ones[k]] == 1:
                    num_neigh_pairs += 1

        # checks if the sign corresponds to (-1)^(number of neighbouring pairs with qubits in 1)
        # tries to avoid having to use complex numbers
        if check_no_cmplx:
            if np.sign(this_ampl) != ref_phase * self_loops_offset * ((-1) ** num_neigh_pairs):
                if print_error:
                    print('Found a minus sign in the wrong position')
                return False
        else:  # things get a little bit nastier if complex global phases are involved
            if np.abs(
                    np.exp(np.angle(this_ampl) * 1.j) - ref_phase * self_loops_offset * ((-1) ** num_neigh_pairs)) > imaginary_part_cutoff:
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

    adj_matrix = find_adj_matrix(state, num_qbts=nqbts)
    test_graphstate = check_correct_signs(state, adj_matrix, num_qbts=nqbts, print_error=print_error)

    if test_graphstate:
        return True, adj_matrix
    else:
        return False, []


if __name__ == '__main__':
    mystate = np.array([1, 1, 1, -1, 1, 1, -1, 1])  # three qubit line
    mystate = np.array([1, 1, 1, -1, -1, -1, 1, -1])  # three qubit line - with loop on last qubit
    # mystate = np.array([1, 1, 1, -1,  1, -1, -1, -1]) # three qubit cycle

    # mystate = np.array([1, 1, 1, -1, 1, -1, -1, -1])
    # mystate = [0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j, # five qubit line
    #           0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #           0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,
    #          -0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j, -0.17677668+0.j,
    #           0.17677668+0.j,  0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,
    #           0.17677668+0.j,  0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #          -0.17677668+0.j, -0.17677668+0.j, -0.17677668+0.j,  0.17677668-0.j,
    #           0.17677668-0.j,  0.17677668-0.j, -0.17677668+0.j,  0.17677668-0.j]

    test_result, Amat = vector_is_graphstate(mystate, print_error=True)

    print(test_result)
    print(Amat)

    if test_result:
        from CodesFunctions.GraphStateClass import GraphState
        import networkx as nx
        import matplotlib.pyplot as plt

        graph = nx.from_numpy_matrix(Amat)
        gstate = GraphState(graph)
        gstate.image()
        plt.show()