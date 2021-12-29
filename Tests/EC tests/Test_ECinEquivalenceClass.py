from CodesFunctions.local_complementation import lc_equivalence_class
from ErrorCorrectionFunctions.EC_DecoderClasses import ind_meas_EC_decoder, log_op_error_prob_from_lookup_dict, \
    teleportation_EC_decoder, teleport_error_prob_from_lookup_dict

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *

    qubits_num = 5
    in_qubit = 0
    error_rate = 0.05

    # graph0 = gen_ring_graph(qubits_num)
    graph0 = gen_random_connected_graph(qubits_num)

    gstate0 = GraphState(graph0)
    gstate0.image(input_qubits=[in_qubit])
    plt.show()

    # The graph states considered is the equivalence class of graph0 (up to graph isomorphism considering the input)
    graphs = lc_equivalence_class(graph0, fixed_node=in_qubit)
    graph_states = [GraphState(this_graph) for this_graph in graphs]
    num_graphs = len(graph_states)

    ########     Plot codes     #########

    print("\nFound", num_graphs, "graphs in the equivalence class")

    print_points = 100
    print_density = int(num_graphs / print_points) + 1

    ####################################################################################
    ################################### PAULI MEASUREMENT ##############################
    ####################################################################################

    # EC_list_X = []
    # EC_list_Y = []
    # EC_list_Z = []
    #
    # for gstate_ix, this_gstate in enumerate(graph_states):
    #     if gstate_ix % print_density == 0:
    #         print('Doing graph', gstate_ix, 'of', num_graphs)
    #     EC_list_X.append(log_op_error_prob_from_lookup_dict(ind_meas_EC_decoder(this_gstate, 'X', in_qubit), error_rate))
    #     EC_list_Y.append(log_op_error_prob_from_lookup_dict(ind_meas_EC_decoder(this_gstate, 'Y', in_qubit), error_rate))
    #     EC_list_Z.append(log_op_error_prob_from_lookup_dict(ind_meas_EC_decoder(this_gstate, 'Z', in_qubit), error_rate))
    #
    # ########     Plot results     #########
    #
    # print(EC_list_X)
    # n, bins, patches = plt.hist(EC_list_X, alpha=1, label='X')
    # bin_size = bins[1]-bins[0]
    # n1, _, _ = plt.hist(np.array(EC_list_Y)+bin_size/3., alpha=0.6, label='Y')
    # n2, _, _ = plt.hist(np.array(EC_list_Z)-bin_size/3, alpha=0.4, label='Z')
    # plt.vlines(error_rate, 0, max(max(n), max(n1), max(n2)), colors='black', linestyles='dashed')
    # populated_bins = [i for i, x in enumerate(n) if x]
    # plt.xlim(min(error_rate, bins[min(populated_bins)])-bin_size, bins[max(populated_bins)]+bin_size)
    # # plt.xlim(0, 0.2)
    # plt.xlabel('Logical error rate')
    # plt.ylabel('Occurrences in local-equivalence class')
    # plt.title('EC phys. error rate: '+str(error_rate))
    # plt.legend()
    # plt.show()

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

    ################################################################################
    ################################### TELEPORTATION ##############################
    ################################################################################

    EC_list = []

    for gstate_ix, this_gstate in enumerate(graph_states):
        if gstate_ix % print_density == 0:
            print('Doing graph', gstate_ix, 'of', num_graphs)
        EC_list.append(
            teleport_error_prob_from_lookup_dict(teleportation_EC_decoder(this_gstate, in_qubit), error_rate))

    ########     Plot results     #########

    print(EC_list)
    n, bins, patches = plt.hist(EC_list, alpha=1, label='Full')
    bin_size = bins[1] - bins[0]
    plt.vlines(error_rate, 0, max(n), colors='black', linestyles='dashed')
    populated_bins = [i for i, x in enumerate(n) if x]
    plt.xlim(min(error_rate, bins[min(populated_bins)]) - bin_size, bins[max(populated_bins)] + bin_size)
    # plt.xlim(0, 0.2)
    plt.xlabel('Teleportation error rate')
    plt.ylabel('Occurrences in local-equivalence class')
    plt.title('EC phys. error rate: ' + str(error_rate))
    plt.legend()
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
