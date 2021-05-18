from CodesFunctions.local_complementation import lc_equivalence_class
from LossToleranceFunctions.LT_qubitencoding import get_possible_stabs_meas, succ_prob_MCestimate, \
    trasmission_scan_MCestimate

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *


    #############################################################################################
    ########  TEST LOSS_TOLERANCE IN A LC CLASS  - LT distribution at single loss value #########
    #############################################################################################

    qubits_num = 8
    in_qubit = 0
    transmission = 0.9
    MC_samples = 1000

    tot_num_graphs = 1000

    le_class_number_list = []
    LT_list = []

    print_points = 100
    print_density = int(tot_num_graphs/print_points) + 1

    for graph_ix in range(tot_num_graphs):
        if graph_ix%print_density == 0:
            print('Doing graph', graph_ix, 'of', tot_num_graphs)
        graph0 = gen_random_connected_graph(qubits_num)
        if nx.is_connected(graph0):
            le_class_number_list.append(len(lc_equivalence_class(graph0, fixed_node=in_qubit)))
            LT_list.append(succ_prob_MCestimate(get_possible_stabs_meas(GraphState(graph0), in_qubit), transmission, MC_samples, in_qubit))

    ########     Plot results     #########

    plt.plot(le_class_number_list, LT_list, 'o', alpha=0.2)
    plt.hlines(transmission, min(le_class_number_list), max(le_class_number_list), colors='black', linestyles='dashed')
    plt.xlabel('Locally-equivalence class cardinality')
    plt.ylabel('LT success probability')
    plt.title('LT tests at transmittivity: '+str(transmission))
    plt.show()
