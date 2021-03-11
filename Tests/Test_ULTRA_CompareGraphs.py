from CodesFunctions.LTCodeClass import LTCode
from CodesFunctions.graphs import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    ##################################
    #####     INITIALISATION     #####
    ##################################
    ##### Define graph dimensions & calculation parameters

    max_m = 5.
    num_MC_trials = 1000
    loss_prob_list = np.linspace(0, 1, 10)

    ##### initialise lists

    graph_codes = []
    labels = []

    ###########################################
    #####     GRAPH CODES DEFINITIONS     #####
    ###########################################

    ########## Square-lattice graph encoding
    nrows = 4
    nlayers = 1
    encode_graph = gen_square_lattice_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = []
    graph_label = 'Square '+str(nrows)+'x'+str(nlayers)

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)

    ########## Hexagonal-lattice graph encoding
    nrows = 4
    nlayers = 1
    encode_graph = gen_hexagonal_lattice_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = []
    graph_label = 'Hexagonal '+str(nrows)+'x'+str(nlayers)

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)

    ########## Crazy-graph encoding
    nrows = 4
    nlayers = 1
    encode_graph = gen_crazy_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = []
    graph_label = 'CrazyGraph '+str(nrows)+'x'+str(nlayers)

    graph_codes.append(LTCode(encode_graph, in_nodes, out_nodes))
    labels.append(graph_label)


    ##########################################
    ### PERFORM ULTRA LOSS TOLERANCE TESTS ###
    ##########################################
    codes_LT_probs_list = []
    for code_ix, this_code in enumerate(graph_codes):
        print("Performing tests on code:", labels[code_ix])
        start_time = time.time()

        tele_succ_probs = this_code.ULTRA_teleport_prob_MC_estimation(loss_prob_list, max_m, num_MC_trials)
        end_time = time.time()
        print("Completed! Time used:", end_time - start_time, "s")
        codes_LT_probs_list.append(tele_succ_probs)

    ####################################
    ########     PLOT DATA     #########
    ####################################

    plt.figure()
    for code_ix, tele_succ_prob_list in enumerate(codes_LT_probs_list):
        plt.plot(loss_prob_list, tele_succ_prob_list, label=labels[code_ix])
    plt.plot(loss_prob_list, 1-np.array(loss_prob_list), 'k--')
    plt.legend(loc='upper right')
    plt.xlabel('Loss-per-photon probability')
    plt.ylabel('Teleportation Probability')
    plt.title('ULTRA Teleportation')
    plt.grid(alpha=0.4, linestyle='--')
    plt.show()

    # plt.figure()
    # mycode.image(with_labels=True)
    # plt.show()
