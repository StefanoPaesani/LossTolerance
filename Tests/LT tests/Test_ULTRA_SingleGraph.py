from CodesFunctions.LTCodeClass import LTCode
from CodesFunctions.graphs import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    ########## Crazy-graph encoding
    nrows = 4
    nlayers = 1
    # encode_graph = gen_crazy_graph(nrows, nlayers)
    encode_graph = gen_square_lattice_graph(nrows, nlayers)
    # encode_graph = gen_triangular_lattice_graph(nrows, nlayers)
    # encode_graph = gen_hexagonal_lattice_graph(nrows, nlayers)
    # encode_graph = gen_multiwire_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    out_nodes = []

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

    #############################################
    ### Monte-Carlo Loss-Tolerance Estimation ###
    #############################################

    num_MC_trials = 100
    max_m = 5.

    print("Starting ULTRA teleportation probability calculation - scans")
    start_time = time.time()

    loss_prob_list = np.linspace(0, 1, 10)
    tele_succ_prob_list = mycode.ULTRA_teleport_prob_MC_estimation(loss_prob_list, max_m, num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    plt.figure()
    plt.plot(loss_prob_list, tele_succ_prob_list)
    plt.show()

    plt.figure()
    mycode.image(with_labels=True)
    plt.show()