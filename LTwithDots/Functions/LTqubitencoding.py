from CodesFunctions.GraphStateClass import GraphState

import qecc as q
import numpy as np
import networkx as nx

from heapq import nlargest

from CodesFunctions.graphs import *


##################################
### STABS GENERATING FUNCTIONS ###
##################################

def get_possible_stabs_meas(gstate, in_qubit=0):
    num_qubits = len(gstate)
    # print("num_qubits", num_qubits)

    # get all possible 2^N stabilizers.
    # TODO: to be improved checking only "smart" stabilizers.
    all_stabs = list(q.from_generators(gstate.stab_gens))

    poss_stabs_list = []
    for stab_ix1 in range(1, (2 ** num_qubits) - 1):
        for stab_ix2 in range(stab_ix1, 2 ** num_qubits):
            stab1 = all_stabs[stab_ix1].op
            stab2 = all_stabs[stab_ix2].op
            ## checks which qubits have anticommuting Paulis
            anticomm_qbts = [qbt for qbt in range(num_qubits) if single_qubit_commute(stab1, stab2, qbt)]

            ## checks that there are exactly two qubits with anticommuting Paulis: the input and the output
            if len(anticomm_qbts) == 2 and in_qubit in anticomm_qbts:
                measurement = [stab1[qbt] if stab1[qbt] is not 'I' else stab2[qbt] for qbt in range(num_qubits)]
                other_meas_qubits = [qbt for qbt in range(num_qubits)
                                     if measurement[qbt] is not 'I' and qbt not in anticomm_qbts]
                meas_weight = num_qubits - measurement.count('I')
                poss_stabs_list.append([anticomm_qbts, other_meas_qubits, [stab1, stab2], measurement, meas_weight])
                # print(stab1, stab2, anticomm_qbts, other_meas_qubits, measurement, meas_weight)

    poss_stabs_list.sort(key=lambda x: x[4])

    return poss_stabs_list


###################################
##### MONTE CARLO DECODING TEST ###
###################################

def MC_decoding(poss_stabs_list, transmission, in_qubit=0, printing=False):
    # Initialize which qubits have been successfully or insuccessfully measured
    meas_out = False  # tracks if the output qubit for current strategy has already been measured or not
    out_qubit_ix = in_qubit  # tracks the index of the current output qubit
    measured_qubits = []  # tracks qubits that have been measured
    lost_qubits = []  # tracks qubits that have been lost
    on_track = True  # tracks if we're still on route to succeding or if we have failed
    finished = False  # tracks if the decoding process has finished or not

    new_strategy = True  # tracks if the current measurement strategy needs to be changed

    while not finished:
        if printing:
            print()
            print("starting new measurement, meas_out", meas_out, ", measured_qubits", measured_qubits, ", lost_qubits", lost_qubits)
            print("Current stabs:", poss_stabs_list)

        # if there are no possible measurement to do, we have failed and we stop
        if len(poss_stabs_list) == 0:
            on_track = False
            finished = True
            # print("failing")
        else:

            if new_strategy:
                # decide new strategy
                meas_config = poss_stabs_list[0]
                temp_out_qb_ix = meas_config[0][1]
                if temp_out_qb_ix != out_qubit_ix:
                    meas_out = False
                    out_qubit_ix = temp_out_qb_ix

                ### Decide which nodes are better to keep as outputs
                poss_outs_dict = {}
                for strat in poss_stabs_list:
                    this_out = strat[0][1]
                    weight = strat[4]
                    add_to_sum = 1. / weight
                    if this_out in poss_outs_dict:
                        poss_outs_dict[this_out] += add_to_sum
                    else:
                        poss_outs_dict[this_out] = add_to_sum
                # hack to have larger indexes first, to facilitate their selection for first in the strategy
                poss_outs_dict = dict(reversed(list(poss_outs_dict.items())))
                # sort the possible outputs from the one least likely to get good strategies to the best one
                poss_outs_dict = dict(sorted(poss_outs_dict.items(), key=lambda item: item[1]))
                # print("poss_outs", poss_outs)

                new_strategy = False

            if printing:
                print("testing measurement meas_config", meas_config)

            if not meas_out:
                # start strategy by trying to measure the output qubit
                meas_qubit_ix = meas_config[0][1]
                if printing:
                    print("measuring out qubit", meas_qubit_ix)
            else:
                # try to measure a non-output qubit
                qubits_to_measure = [x for x in meas_config[1] if x not in measured_qubits]
                if len(qubits_to_measure) == 0:
                    if printing:
                        print("SUCCEDING: no more qubits to measure")
                    # If there are no more qubits to measure, we've succeded and we stop.
                    finished = True
                else:
                    # check if there are some 'safe' options that are not in the possible outputs list
                    qubits_to_meas_nooutputs = [x for x in qubits_to_measure if x not in poss_outs_dict]
                    if len(qubits_to_meas_nooutputs) > 0:
                        # if there are safe options, pick one of these (starting from the largest one, inspired by trees)
                        meas_qubit_ix = qubits_to_meas_nooutputs[-1]
                    else:
                        # if not, pick the possible first in qubits_to_meas in order of preference of the outputs
                        qubits_to_meas = [x for x in poss_outs_dict if x in qubits_to_measure]
                        meas_qubit_ix = qubits_to_meas[0]

                    if printing:
                        print("measuring qubit", meas_qubit_ix)

            if not finished:
                # test if the measured photon is alive
                is_alive = (np.random.rand() < transmission)
                if is_alive:
                    if printing:
                        print("qubit is ALIVE")
                    measured_qubits.append(meas_qubit_ix)
                    if meas_out:
                        # if the measured qubit is not an output, it was measured in a fixed basis, and we update the possible measurement strategies accordingly
                        poss_stabs_list = filter_measured_qubit_fixed_basis(poss_stabs_list, meas_config, meas_qubit_ix)
                    else:
                        # if the measured qubit is an output, it was measured in an arbitrary basis, and we update the possible measurement strategies accordingly
                        poss_stabs_list = filter_measured_qubit_output_basis(poss_stabs_list, meas_config, meas_qubit_ix)
                        meas_out = True

                else:
                    if printing:
                        print("qubit is LOST")
                    # if the photon is lost, we need to update the measurement strategies accordingly and restart with a new strategy
                    lost_qubits.append(meas_qubit_ix)
                    poss_stabs_list = filter_lost_qubit(poss_stabs_list, meas_qubit_ix)
                    new_strategy = True

    # see if we succeded or failed
    if on_track:
        return True
    else:
        return False


def succ_prob_MCestimate(poss_stabs_list, transmission, MC_samples=1000, in_qubit=0):
    return sum(
        [MC_decoding(poss_stabs_list, transmission, in_qubit) for i in range(MC_samples)]) / MC_samples


def trasmission_scan_MCestimate(poss_stabs_list, transm_samples=21, MC_samples=1000, in_qubit=0, trasm_range=[0, 1]):
    return [succ_prob_MCestimate(poss_stabs_list, this_transm, MC_samples, in_qubit)
            for this_transm in np.linspace(trasm_range[0], trasm_range[1], transm_samples)]


## Function that find the "num_best_graphs" graphs which higher success rate
def find_best_graphs(graph_states, transmission, MC_sims, print_status=True, num_best_graphs=None, in_qubit=0):
    if not num_best_graphs:
        num_best_graphs = int(np.sqrt(len(graph_states)))

    if print_status:
        num_graphs = len(graph_states)
        succ_prob_for_graphs_list = []
        for graph_ix, this_gstate in enumerate(graph_states):
            if (graph_ix % 100) == 0:
                print("Testing graph", graph_ix, 'of', num_graphs)
            succ_prob_for_graphs_list.append(
                succ_prob_MCestimate(get_possible_stabs_meas(this_gstate, in_qubit), transmission, MC_sims, in_qubit))
    else:
        succ_prob_for_graphs_list = [
            succ_prob_MCestimate(get_possible_stabs_meas(this_gstate, in_qubit), transmission, MC_sims, in_qubit) for
            this_gstate in graph_states]

    ##  Find best codes

    best_probs = nlargest(num_best_graphs, succ_prob_for_graphs_list)
    min_best_prob = min(best_probs)

    return [this_gstate for graph_ix, this_gstate in enumerate(graph_states)
            if succ_prob_for_graphs_list[graph_ix] >= min_best_prob]


##############################
### OTHER USEFUL FUNCTIONS ###
##############################

def single_qubit_commute(pauli1, pauli2, qbt):
    """
    Returns 0 if the operators on the qbt-th qubit of the two operators in the Pauli group commute,
    and 1 if they anticommute.
    """
    if pauli1[qbt] == 'I' or pauli2[qbt] == 'I' or pauli1[qbt] == pauli2[qbt]:
        return 0
    else:
        return 1


# keeps only the possible strategies in which the lost_qbt_ix is allowed to be lost
def filter_lost_qubit(poss_stabs_list, lost_qbt_ix):
    return [these_stabs for these_stabs in poss_stabs_list if these_stabs[3][lost_qbt_ix] == 'I']


# keeps only the possible strategies in which the meas_qbt_ix is not an output and have it in the fixed basis measured
# (for qubits measured in fixed basis that cannot anymore be used as outputs or measured in other bases)
def filter_measured_qubit_fixed_basis(poss_stabs_list, meas_config, meas_qbt_ix):
    fixed_basis = meas_config[3][meas_qbt_ix]
    return [these_stabs for these_stabs in poss_stabs_list if
            (these_stabs[0][1] != meas_qbt_ix and these_stabs[3][meas_qbt_ix] == fixed_basis)]

# keeps only the possible strategies in which the meas_qbt_ix is an output or it is not measured
# (for qubits measured in an arbitrary output basis that cannot anymore be measured in fixed bases)
def filter_measured_qubit_output_basis(poss_stabs_list, meas_config, meas_qbt_ix):
    return [these_stabs for these_stabs in poss_stabs_list if
            (these_stabs[0][1] == meas_qbt_ix or these_stabs[3][meas_qbt_ix] == 'I')]


# Theoretical success probability for tree graphs from Phys. Rev. Lett. 97, 120501 (2006)
def p_analyt_tree(t, branch_list):
    m = len(branch_list) - 1
    loss = 1 - t
    R1 = get_R(1, loss, branch_list, m)
    R2 = get_R(2, loss, branch_list, m)
    return (((1 - loss + loss * R1) ** branch_list[0]) - ((loss * R1) ** branch_list[0])) * (
            (1 - loss + loss * R2) ** branch_list[1])


def get_R(k, loss, branch_list, m):
    if k > m:
        return 0
    else:
        if k == m:
            temp_b = 0
        else:
            temp_b = branch_list[k + 1]
        this_R = get_R(k + 2, loss, branch_list, m)
        return 1 - ((1 - (1 - loss) * ((1 - loss + loss * this_R) ** temp_b)) ** branch_list[k])


# quick graph state definition
def graphstate_from_nodes_and_edges(graph_nodes, graph_edges):
    graph = nx.Graph()
    graph.add_nodes_from(graph_nodes)
    graph.add_edges_from(graph_edges)
    return GraphState(graph)


########################################################################################################################
##############################
###          MAIN          ###
##############################


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.LTCodeClass import powerset

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    # three graph
    # branching = [2, 1]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    ### fully connected graph
    # graph = gen_fullyconnected_graph(7)
    # gstate = GraphState(graph)

    ### ring graph
    graph = gen_ring_graph(5)
    gstate = GraphState(graph)

    ## get list of possible measurements to encode & decode the state
    poss_stabs_list = get_possible_stabs_meas(gstate, in_qubit)

    ##############################################################################
    ################################### SINGLE TEST ##############################
    ##############################################################################

    ## define channel transmission
    # transmission = 0.7
    # decoding_succ = MC_decoding(poss_stabs_list, transmission, in_qubit, printing=True)
    #
    # ## see if we succeded or failed
    # if decoding_succ:
    #     print("Succeded :)")
    # else:
    #     print("Failed :(")

    ##################################################################
    ################# TRANSMISSION SCAN ##############################
    ##################################################################

    eff_list_num = 21
    MC_sims = 1000

    succ_prob_list = trasmission_scan_MCestimate(poss_stabs_list, eff_list_num, MC_sims, in_qubit)

    eff_list = np.linspace(0, 1, eff_list_num)
    plt.plot(eff_list, succ_prob_list, label='LT qubit')
    # plt.plot(eff_list, [p_analyt_tree(t, branching) for t in eff_list], 'b:', linewidth=2, label='LT tree-theo')
    plt.plot(eff_list, eff_list, 'k-', label='direct transm.')
    plt.legend()
    plt.show()

    plt.subplot()
    gstate.image(with_labels=True)
    plt.show()

    #######################################################################################
    ################# TRANSMISSION SCAN - ALL N-Qubit graphs ##############################
    #######################################################################################

    # qubits_num = 5
    #
    # graph_nodes = list(range(qubits_num))
    # all_possible_edges = combinations(graph_nodes, 2)
    # all_graphs_by_edges = list(powerset(all_possible_edges))
    # num_graphs = len(all_graphs_by_edges)
    #
    # graph_states = {}
    # labels = []
    # succ_prob_scan_list = []
    #
    # eff_list_num = 21
    # MC_sims = 100
    # in_qubit = 0
    #
    # for edges_conf_ix, graph_edges in enumerate(all_graphs_by_edges):
    #     print('Testing graph', edges_conf_ix, 'of', num_graphs)
    #     graph = nx.Graph()
    #     graph.add_nodes_from(graph_nodes)
    #     graph.add_edges_from(graph_edges)
    #     this_gstate = GraphState(graph)
    #
    #     graph_states[(edges_conf_ix)] = this_gstate
    #
    #     labels.append(str(edges_conf_ix))
    #
    #     poss_stabs_list = get_possible_stabs_meas(this_gstate, in_qubit)
    #
    #     succ_prob_scan_list.append(trasmission_scan_MCestimate(poss_stabs_list, eff_list_num, MC_sims, in_qubit))
    #
    # for this_prob_scan in succ_prob_scan_list:
    #     plt.plot(eff_list, this_prob_scan)
    # plt.plot(eff_list, eff_list, 'k-')
    # plt.show()
    #

    #######################################################################
    ################# SEARCH FOR BEST GRAPHS ##############################
    #######################################################################

    # qubits_num = 5
    #
    # graph_nodes = list(range(qubits_num))
    # all_possible_edges = combinations(graph_nodes, 2)
    # all_graphs_by_edges = list(powerset(all_possible_edges))
    # num_graphs = len(all_graphs_by_edges)
    #
    # graph_states = [graphstate_from_nodes_and_edges(graph_nodes, these_edges) for these_edges in all_graphs_by_edges]
    #
    # in_qubit = 0
    #
    # ########   Round 1  #########
    #
    # MC_sims = 1000
    # transmission = 0.9
    # num_best_codes = int(np.sqrt(num_graphs))
    # print_status = True
    #
    # print("\nStarting round 1")
    # graph_states = find_best_graphs(graph_states, transmission, MC_sims, print_status, num_best_codes, in_qubit)
    #
    # ########   Round 2  #########
    #
    # MC_sims = 5000
    # transmission = 0.9
    # num_best_codes = min([20, len(graph_states)])
    # print_status = False
    #
    # print("\nStarting round 2")
    # graph_states = find_best_graphs(graph_states, transmission, MC_sims, print_status, num_best_codes, in_qubit)
    #
    # ########     PLOT BEST CODES     #########
    # best_codes = graph_states
    # codes_labels = list(range(len(best_codes)))
    #
    # print("\nDoing plots")
    #
    # # plot best codes transmission scans
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
    # n = num_best_codes
    # i = 2
    # while i * i < n:
    #     while n % i == 0:
    #         n = n / i
    #     i = i + 1
    #
    # n_plot_rows = num_best_codes / n
    # n_plot_cols = n
    #
    # for code_ix in range(num_best_codes):
    #     plt.subplot(n_plot_rows, n_plot_cols, code_ix + 1)
    #     best_codes[code_ix].image(with_labels=True)
    # plt.show()