from CodesFunctions.LTCodeClass import LTCode
from CodesFunctions.graphs import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    ########## Crazy-graph encoding
    nrows = 4
    nlayers = 4
    # encode_graph = gen_crazy_graph(nrows, nlayers)
    # encode_graph = gen_square_lattice_graph(nrows, nlayers)
    # encode_graph = gen_triangular_lattice_graph(nrows, nlayers)
    encode_graph = gen_hexagonal_lattice_graph(nrows, nlayers)
    # encode_graph = gen_multiwire_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))

    # in_nodes = [[0, 1, 2], [3, 4, 5]]
    # out_nodes = [[6, 7, 8], [9, 10]]
    # in_nodes = [[0, 1]]
    # out_nodes = [[0, 1]]
    # in_nodes = [[0, 1], [2]]
    # out_nodes = [[0, 1], [2]]
    # in_nodes = [[0, 1], [2, 3]]
    # out_nodes = [[4, 5], [6, 7]]
    # in_nodes = [[0, 1, 2], [3, 4, 5]]
    # out_nodes = [[6, 7, 8], [9, 10, 11]]

    ########## gen_fullyconnected_graph
    # nqbts = 11
    # encode_graph = gen_linear_graph(nqbts)
    # encode_graph = gen_fullyconnected_graph(nqbts)
    # encode_graph = gen_ring_graph(nqbts)
    # in_nodes = list(range(int(nqbts / 2)))
    # out_nodes = list(range(int(nqbts / 2), nqbts))

    ##################
    ### START TEST ###
    ##################

    mycode = LTCode(encode_graph, in_nodes, out_nodes)

    start_time = time.time()
    alpha = 2
    tele_meas_SPalg = mycode.SPalgorithm_valid_teleportation_meas(max_m_increase=alpha, test_inouts=True,
                                                                  exclude_input_ys=True, return_evolution=False)
    end_time = time.time()
    print("Time passed for SP algorithm with alpha", alpha, ":", end_time - start_time, "s, found", len(tele_meas_SPalg)
          , "measurements")

    #############################################
    ### Monte-Carlo Loss-Tolerance Estimation ###
    #############################################

    loss_prob = 0.2
    num_MC_trials = 100

    weight_fact = 4
    max_tree_depth = 0

    print("Starting unheralded loss tolerance probability calculation - single loss values")
    start_time = time.time()
    teleport_succ_prob = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob,
                                                                            follow_curr_best=False,
                                                                            weight_fact=weight_fact,
                                                                            max_tree_depth=max_tree_depth,
                                                                            MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)
    print("Unheralded loss teleport succ prob:", teleport_succ_prob)

    print("Starting unheralded loss tolerance probability calculation - scans")
    start_time = time.time()

    loss_prob_list = np.linspace(0, 1, 10)
    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=False,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    plt.figure()
    plt.plot(loss_prob_list, tele_succ_prob_list)
    plt.show()

    plt.figure()
    mycode.image(with_labels=True)
    plt.show()
