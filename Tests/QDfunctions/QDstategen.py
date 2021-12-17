from CodesFunctions.vector_is_graph import vector_is_graphstate
from CodesFunctions.GraphStateClass import GraphState

import cirq
import networkx as nx
from itertools import product
import numpy as np

# Allowed_Gates = ['I', 'H', 'X', 'Z', 'Y', 'SX', 'SZ', 'T']
Allowed_Gates = ['H', 'SX', 'X', 'I']
# Allowed_Gates = ['H', 'X', 'I']

H_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I_mat = np.identity(2)


def simulate_qd_scheme(spin_gates_list, print_circuit=False):
    """QDfunctions that calculates the state vectors of the QD spin (0-th qubit) and of the photonic qubits after the
    pulse sequence. spin_gates_list. Returns the state vector. If print_circuit is True, the simulated circuit is
    printed.

    :param list spin_gates_list: gates to be done on the QD after each round of photon generation.
    """

    for this_rot in spin_gates_list:
        if this_rot not in Allowed_Gates:
            raise ValueError("All rotation gates need to be in allowed list.")

    num_phot = len(spin_gates_list)
    qubits = cirq.LineQubit.range(num_phot + 1)

    # start preparing spin in |+>
    all_gates = [cirq.H(qubits[0])]

    for phot_ix, this_rot in enumerate(spin_gates_list):
        # perform CNOT (photon generation)
        all_gates.append(cirq.CNOT(qubits[0], qubits[phot_ix + 1]))
        # spin ends up in opposite spin after the pi gate
        all_gates.append(cirq.X(qubits[0]))
        # do rotation on spin
        if this_rot == 'H':
            all_gates.append(cirq.H(qubits[0]))

        if this_rot == 'X':
            all_gates.append(cirq.X(qubits[0]))
        if this_rot == 'Y':
            all_gates.append(cirq.Y(qubits[0]))
        if this_rot == 'Z':
            all_gates.append(cirq.Z(qubits[0]))
        if this_rot == 'SX':
            all_gates.append(cirq.H(qubits[0]))
            all_gates.append(cirq.S(qubits[0]))
            all_gates.append(cirq.H(qubits[0]))
        if this_rot == 'SZ':
            all_gates.append(cirq.S(qubits[0]))
        if this_rot == 'T':
            all_gates.append(cirq.T(qubits[0]))

    circuit = cirq.Circuit(all_gates)

    if print_circuit:
        print(circuit)

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)

    return result.final_state_vector


def get_singlequbitU_total_matr(n, single_qubit_U, num_qbts):
    temp_mat = np.identity(2 ** n)
    temp_mat = np.kron(temp_mat, single_qubit_U)
    temp_mat = np.kron(temp_mat, np.identity(2 ** (num_qbts - n - 1)))
    return temp_mat


def hadamards_to_uniform(state, num_qbts=None):
    """QDfunctions that, if a state doe not have uniform amplitudes, tries to apply Hadamards to obtain a uniform
    superposition. Assumes that the amplitude for |00..00> is nonzero.
    """

    if num_qbts:
        nqbts = num_qbts
    else:
        nqbts = int(np.log2(len(state)))
    applied_hadamard = np.zeros(nqbts)

    # if amplitudes are all the same already, return the state as it is.
    if np.all(np.abs(state) == np.abs(state[0])):
        return state, applied_hadamard

    # if state[0] == 0:
    #     raise ValueError('Amplitude for |00..00> must be nonzero to use the Hadamard uniformicator')

    for qb_ix in range(num_qbts):
        # print('\nqb_ix', qb_ix, 'ampl:', state[2 ** (num_qbts - qb_ix - 1)])
        if state[2 ** (num_qbts - qb_ix - 1)] == 0:
            applied_hadamard[qb_ix] = 1
            state = get_singlequbitU_total_matr(qb_ix, H_mat, nqbts) @ state
            # print(state)
    # print('state_with_H:', state)
    # print('H_list:', applied_hadamard)
    return state, applied_hadamard


def does_qd_give_graph(spin_gates_list, accept_hadamards=True, print_error=False):
    """QDfunctions that checks if the qd and the associated pulse sequence generates a graph state. If it does,
    it also returns the graph state as a GraphState class element.

    :param list spin_gates_list: gates to be done on the QD after each round of photon generation.
    """
    tot_qubits_num = len(spin_gates_list) + 1
    state_vector = simulate_qd_scheme(spin_gates_list, print_circuit=False)
    # print('Initial state_vec:', state_vector)
    applied_hadamard = []
    if accept_hadamards:
        state_vector, applied_hadamard = hadamards_to_uniform(state_vector, num_qbts=tot_qubits_num)
        # print('State_vec after hadamards:', state_vector)
    is_graph, adj_mat, local_phases = vector_is_graphstate(state_vector, num_qbts=tot_qubits_num,
                                                           print_error=print_error)
    if is_graph:
        return True, adj_mat, applied_hadamard, local_phases
    else:
        return False, [], applied_hadamard, []


###################   Miscelannea functions ######################################
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


###################   Test functions ######################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from LossToleranceFunctions.LT_qubitencoding import get_possible_stabs_meas, trasmission_scan_MCestimate, \
        find_best_graphs
    from CodesFunctions.local_complementation import lc_equivalence_class, check_isomorphism_with_fixednode

    #############################################
    ####### TEST A SINGLE PULSE SEQUENCE ########
    #############################################

    # list of rotations to be done on the spin
    # rots_list = ['X', 'X', 'X', 'X'] # generate GHZ states
    # rots_list = ['H', 'H', 'H', 'X'] # generate a lines
    #
    # #### Calculates generated state vector
    # final_state = simulate_qd_scheme(rots_list, print_circuit=True)
    # print(final_state)
    # ###### Check if state is graph state
    # is_graph, adj_mat, applied_hadamard, local_phases = does_qd_give_graph(rots_list, accept_hadamards=True, print_error=True)
    # if is_graph:
    #     gstate = GraphState(nx.from_numpy_matrix(adj_mat))
    #     print("Got a graph! Rotation sequence:", rots_list, " Local Z phases:", local_phases, " Applied hadamards: ",
    #           applied_hadamard)
    #     plt.subplot()
    #     gstate.image(with_labels=True)
    #     plt.show()
    # else:
    #     print("Not a graph :(")
    #
    # ### Test the loss-tolerance of the generated graph
    # if is_graph:
    #     transm_sampl_num = 11
    #     MC_samples = 1000
    #
    #     poss_stabs = get_possible_stabs_meas(gstate)
    #     trasm_scan_list = trasmission_scan_MCestimate(poss_stabs, transm_sampl_num, MC_samples)
    #
    #     transm_list = np.linspace(0, 1, transm_sampl_num)
    #
    #     plt.plot(transm_list, trasm_scan_list, label='encoded')
    #     plt.plot(transm_list, transm_list, 'k:', label='direct')
    #     plt.legend()
    #     plt.show()

    ########################################################
    ####### TEST ALL PULSE SEQUENCES WITH N PHOTONS ########
    ########################################################

    num_phots = 5

    include_lc = True
    input_qubit = 0

    all_rots_lists = product(Allowed_Gates, repeat=num_phots)

    obt_graphs = []
    used_rots = []
    lc_class_representatives = []

    ## DELETE
    step = 0

    for rots_list in all_rots_lists:
        # print('rots_list', rots_list)
        is_graph, adj_mat, applied_hadamard, local_phases = does_qd_give_graph(rots_list)

        if is_graph:
            # print('FOUND NEW GRAPHS!')
            graph = nx.from_numpy_matrix(adj_mat)
            np.fill_diagonal(adj_mat, 0)

            if include_lc:
                if not check_isomorphism_with_fixednode(graph, lc_class_representatives, fixed_node=input_qubit):
                    lc_class_representatives.append(graph)
                    full_new_lc_class = lc_equivalence_class(graph, fixed_node=input_qubit)
                    obt_graphs = obt_graphs + full_new_lc_class
                    used_rots = used_rots + [rots_list for i in range(len(full_new_lc_class))]
                    # print('Total class representatives number:', len(lc_class_representatives))
                    # print('Total graph number:', len(obt_graphs))
            else:
                if not arreq_in_list(adj_mat, obt_graphs):
                    # print("Got a NEW graph! Rotation sequence:", rots_list, " Local Z phases:", local_phases, " Applied hadamards: ", applied_hadamard)
                    print(rots_list, local_phases, applied_hadamard)
                    obt_graphs.append(adj_mat)
                    used_rots.append(rots_list)
                    plt.subplot()
                    gstate = GraphState(nx.from_numpy_matrix(adj_mat))
                    gstate.image(with_labels=True)
                    plt.show()
    if not include_lc:
        obt_graphs = [nx.from_numpy_matrix(this_A) for this_A in obt_graphs]


    # plot all obtained graphs
    num_graphs = len(obt_graphs)
    n = int(np.sqrt(num_graphs))

    n_plot_rows = n
    n_plot_cols = num_graphs / n
    if not isinstance(n_plot_cols, int):
        n_plot_cols = int(n_plot_cols) + 1

    # for code_ix in range(num_graphs):
    print('All obtained', num_graphs, 'graphs:')
    for code_ix, this_g in enumerate(obt_graphs):
        plt.subplot(n_plot_rows, n_plot_cols, code_ix + 1)
        GraphState(this_g).image(with_labels=True)
        plt.text(0, 0, str(code_ix)+':\n'+str(used_rots[code_ix]), ha='center')
    plt.show()

    ####################################################
    ####### FIND BEST GRAPHS FOR LOSS TOLERANCE ########
    ####################################################

    # num_phots = 4
    #
    # include_lc = True
    #
    # MC_sims = 1000
    # transmission = 0.9
    # in_qubit = 0
    #
    # num_best_graphs_to_keep = 6
    #
    # ###### Find all graphs ######
    # print("Finding all graphs")
    # all_rots_lists = product(Allowed_Gates, repeat=num_phots)
    # lc_class_representatives = []
    # gen_graphs = []
    # used_rots = []
    # used_adj_mat = []
    # for rots_list in all_rots_lists:
    #     is_graph, adj_mat, applied_hadamard, local_phases = does_qd_give_graph(rots_list)
    #     if is_graph:
    #         graph = nx.from_numpy_matrix(adj_mat)
    #         if include_lc:
    #             if not check_isomorphism_with_fixednode(graph, lc_class_representatives, fixed_node=in_qubit):
    #                 lc_class_representatives.append(graph)
    #                 full_new_lc_class = lc_equivalence_class(graph, fixed_node=in_qubit)
    #                 gen_graphs = gen_graphs + full_new_lc_class
    #                 used_rots = used_rots + [rots_list for i in range(len(full_new_lc_class))]
    #         else:
    #             if not arreq_in_list(adj_mat, used_adj_mat):
    #                 gen_graphs.append(graph)
    #                 used_rots.append(rots_list)
    #                 used_adj_mat.append(adj_mat)
    #
    # graph_states = [GraphState(this_graph) for this_graph in gen_graphs]
    # print("Graph search finished. Number of graphs found:", len(graph_states))
    #
    # ########   find best loss tolerant graphs  #########
    # in_qubit = 0
    #
    # ########   round 1  #########
    # num_best_codes = min([num_best_graphs_to_keep, len(graph_states)])
    # print_status = False
    #
    # print("\nStarting round 1")
    # best_codes = find_best_graphs(graph_states, transmission, MC_sims, print_status, num_best_codes, in_qubit)
    # best_rots = [used_rots[graph_states.index(this_best_code)] for this_best_code in best_codes]
    #
    # ########     PLOT BEST CODES     #########
    # codes_labels = list(range(len(best_codes)))
    #
    # print("\nDoing plots")
    #
    # ########  Plot best codes transmission scans  ########
    # MC_samples = 1000
    # transm_samples = 21
    # transm_list = np.linspace(0, 1, transm_samples)
    #
    # for gstate_ix, this_gstate in enumerate(best_codes):
    #     trasm_scan_list = trasmission_scan_MCestimate(get_possible_stabs_meas(this_gstate, in_qubit), transm_samples,
    #                                                   MC_samples, in_qubit)
    #
    #     if gstate_ix < 10:
    #         plt.plot(transm_list, trasm_scan_list, label=gstate_ix)
    #     else:
    #         plt.plot(transm_list, trasm_scan_list, linestyle='dashed', label=gstate_ix)
    # plt.plot(transm_list, transm_list, 'k:')
    # plt.legend()
    # plt.show()
    #
    # # plot all best graphs
    #
    # n = int(np.sqrt(num_best_codes))
    #
    # n_plot_rows = n
    # n_plot_cols = num_best_codes / n
    # if not isinstance(n_plot_cols, int):
    #     n_plot_cols = int(n_plot_cols) + 1
    #
    # for code_ix in range(num_best_codes):
    #     plt.subplot(n_plot_rows, n_plot_cols, code_ix + 1)
    #     best_codes[code_ix].image(with_labels=True)
    #     plt.text(0, 0, str(code_ix) + ':\n' + str(best_rots[code_ix]), ha='center')
    # plt.show()

