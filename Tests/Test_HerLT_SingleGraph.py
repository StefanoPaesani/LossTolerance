from CodesFunctions.LTCodeClass import LTCode
from CodesFunctions.graphs import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    ########## Crazy-graph encoding
    nrows = 4
    nlayers = 4
    encode_graph = gen_crazy_graph(nrows, nlayers)
    # encode_graph = gen_square_lattice_graph(nrows, nlayers)
    # encode_graph = gen_triangular_lattice_graph(nrows, nlayers)
    # encode_graph = gen_hexagonal_lattice_graph(nrows, nlayers)
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
    alpha = 1
    tele_meas_SPalg = mycode.SPalgorithm_valid_teleportation_meas(max_m_increase=alpha, test_inouts=True,
                                                                  exclude_input_ys=True, return_evolution=False)
    end_time = time.time()
    print("Time passed for SP algorithm with alpha", alpha, ":", end_time - start_time, "s, found", len(tele_meas_SPalg)
          , "measurements")

    #############################################
    ### Monte-Carlo Loss-Tolerance Estimation ###
    #############################################

    loss_prob = 0.5
    num_MC_trials = 1000

    teleport_succ_prob = mycode.Heralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob,
                                                                          MC_trials=num_MC_trials,
                                                                          use_indip_loss_combs=True)
    print("Heralded loss teleport succ prob:", teleport_succ_prob)

    loss_prob_list = np.linspace(0, 1, 50)
    tele_succ_prob_list = mycode.Heralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                           MC_trials=num_MC_trials,
                                                                           use_indip_loss_combs=True)

    plt.figure()
    plt.plot(loss_prob_list, tele_succ_prob_list)
    plt.show()

    plt.figure()
    mycode.image(with_labels=True)
    plt.show()
