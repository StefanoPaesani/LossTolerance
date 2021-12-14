from CodesFunctions.LTCodeClass import LTCode
from CodesFunctions.graphs import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    ##################################
    #####     INITIALISATION     #####
    ##################################
    ##### Define graph dimensions & calculation parameters
    nrows = 4
    nlayers = 4

    alpha = 2

    num_MC_trials = 1000
    loss_prob_list = np.linspace(0, 1, 30)

    ##### initialise lists

    graph_codes = []
    labels = []

    ###########################################
    #####     GRAPH CODES DEFINITIONS     #####
    ###########################################

    ########## Square-lattice graph encoding
    encode_graph = gen_square_lattice_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))
    graph_label = 'Square'

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)

    ########## Triangular-lattice graph encoding
    encode_graph = gen_triangular_lattice_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))
    graph_label = 'Triangular'

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)

    ########## Hexagonal-lattice graph encoding
    encode_graph = gen_hexagonal_lattice_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))
    graph_label = 'Hexagonal'

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)

    ########## Crazy-graph encoding
    encode_graph = gen_crazy_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))
    graph_label = 'Crazy Graph'

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)

    ########## Crazy-graph encoding - Only 1 column
    encode_graph = gen_crazy_graph(nlayers*nrows, 1)
    in_nodes = list(range(nlayers*nrows))
    out_nodes = list(range(nlayers*nrows))
    graph_label = 'Crazy Graph - 1Col'

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    plt.figure()
    graph_codes[-1].image(with_labels=True)
    plt.show()
    labels.append(graph_label)

    ########## Fully-connected-graph encoding
    nqbts = nrows * nlayers
    encode_graph = gen_fullyconnected_graph(nqbts)
    in_nodes = list(range(int(nqbts / 2)))
    out_nodes = list(range(int(nqbts / 2), nqbts))
    graph_label = 'Fully-Conn.'

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)

    ####################################
    ### PERFORM LOSS TOLERANCE TESTS ###
    ####################################
    codes_LT_probs_list = []
    for code_ix, this_code in enumerate(graph_codes):
        print("Performing tests on code:", labels[code_ix])
        start_time = time.time()
        tele_meas = this_code.SPalgorithm_valid_teleportation_meas(max_m_increase=alpha, test_inouts=True,
                                                                   exclude_input_ys=True, return_evolution=False)
        print("Found", len(tele_meas), "measurements")
        tele_succ_probs = this_code.Heralded_loss_teleport_prob_MC_estimation(tele_meas, loss_prob_list,
                                                                              MC_trials=num_MC_trials,
                                                                              use_indip_loss_combs=True)
        end_time = time.time()
        print("Completed! Time used:", end_time - start_time, "s")
        codes_LT_probs_list.append(tele_succ_probs)

    ####################################
    ########     PLOT DATA     #########
    ####################################

    plt.figure()
    for code_ix, tele_succ_prob_list in enumerate(codes_LT_probs_list):
        plt.plot(loss_prob_list, tele_succ_prob_list, label=labels[code_ix])
    plt.legend(loc='upper right')
    plt.xlabel('Loss-per-photon probability')
    plt.ylabel('Teleportation Probability')
    plt.title('One-qubit Teleportation - Heralded Loss')
    plt.grid(alpha=0.4, linestyle='--')
    plt.show()

    # plt.figure()
    # mycode.image(with_labels=True)
    # plt.show()
