from CodesFunctions.local_complementation import lc_equivalence_class
from LossToleranceFunctions.LT_qubitencoding import get_possible_stabs_meas, succ_prob_MCestimate, \
    trasmission_scan_MCestimate

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *


    #############################################################################################
    ########  TEST LOSS_TOLERANCE IN A LC CLASS  - LT distribution at single loss value #########
    #############################################################################################

    qubits_num = 6
    in_qubit = 0
    transmission = 0.9

    # graph0 = gen_ring_graph(qubits_num)
    graph0 = gen_random_connected_graph(qubits_num)

    # The graph states considered is the equivalence class of graph0 (up to graph isomorphism considering the input)
    graphs = lc_equivalence_class(graph0, fixed_node=in_qubit)
    graph_states = [GraphState(this_graph) for this_graph in graphs]
    num_graphs = len(graph_states)

    ########     Plot codes     #########

    print("\nFound", num_graphs, "graphs in the equivalence class")

    # plot best codes transmission scans
    MC_samples = 1000

    print_points = 100
    print_density = int(num_graphs/print_points) + 1

    LT_list=[]
    for gstate_ix, this_gstate in enumerate(graph_states):
        if gstate_ix%print_density == 0:
            print('Doing graph', gstate_ix, 'of', num_graphs)
        LT_list.append(succ_prob_MCestimate(get_possible_stabs_meas(this_gstate, in_qubit), transmission, MC_samples, in_qubit))

    print(LT_list)

    n, bins, patches = plt.hist(LT_list, alpha=0.75)
    plt.vlines(transmission, 0, max(n), colors='black', linestyles='dashed')
    plt.xlim(min([0.6, 0.8]), 1)
    plt.xlim(min([bins[next((i for i, x in enumerate(n) if x), 0)], 0.8]), 1)
    plt.xlabel('LT success probability')
    plt.ylabel('Occurrences in local-equivalence class')
    plt.title('LT tests at transmittivity: '+str(transmission))
    plt.show()

    # plot all best graphs
    if num_graphs < 200:
        n = int(np.sqrt(num_graphs))

        n_plot_rows = n
        n_plot_cols = num_graphs / n
        if not isinstance(n_plot_cols, int):
            n_plot_cols = int(n_plot_cols) + 1

        for code_ix in range(num_graphs):
            plt.subplot(n_plot_rows, n_plot_cols, code_ix + 1)
            graph_states[code_ix].image(with_labels=True, input_qubits=[in_qubit])
        plt.show()


    ###########################################################################
    ########  TEST LOSS_TOLERANCE IN A LC CLASS  - Full transmittence #########
    ###########################################################################

    # qubits_num = 7
    # in_qubit = 0
    #
    # # graph0 = gen_ring_graph(qubits_num)
    # graph0 = gen_random_connected_graph(qubits_num)
    #
    # # The graph states considered is the equivalence class of graph0 (up to graph isomorphism considering the input)
    # graphs = lc_equivalence_class(graph0, fixed_node=in_qubit)
    # graph_states = [GraphState(this_graph) for this_graph in graphs]
    # num_graphs = len(graph_states)
    #
    # ########     Plot codes     #########
    #
    # print("\nFound", num_graphs, "graphs in the equivalence class")
    #
    # # plot best codes transmission scans
    # MC_samples = 1000
    # transm_samples = 11
    # transm_list = np.linspace(0, 1, transm_samples)
    #
    # print_points = 100
    # print_density = int(num_graphs/print_points) + 1
    #
    # for gstate_ix, this_gstate in enumerate(graph_states):
    #     if gstate_ix%print_density == 0:
    #         print('Doing graph', gstate_ix, 'of', num_graphs)
    #     trasm_scan_list = trasmission_scan_MCestimate(get_possible_stabs_meas(this_gstate, in_qubit), transm_samples,
    #                                                   MC_samples, in_qubit)
    #
    #     if gstate_ix < 10:
    #         plt.plot(transm_list, trasm_scan_list, label=gstate_ix)
    #     else:
    #         plt.plot(transm_list, trasm_scan_list, linestyle='dashed', label=gstate_ix)
    # plt.plot(transm_list, transm_list, 'k:')
    # if num_graphs < 20:
    #     plt.legend()
    # plt.show()
    #
    # # plot all best graphs
    # if num_graphs < 200:
    #     n = int(np.sqrt(num_graphs))
    #
    #     n_plot_rows = n
    #     n_plot_cols = num_graphs / n
    #     if not isinstance(n_plot_cols, int):
    #         n_plot_cols = int(n_plot_cols) + 1
    #
    #     for code_ix in range(num_graphs):
    #         plt.subplot(n_plot_rows, n_plot_cols, code_ix + 1)
    #         graph_states[code_ix].image(with_labels=True, input_qubits=[in_qubit])
    #     plt.show()
