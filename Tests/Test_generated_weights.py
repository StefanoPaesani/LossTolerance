from CodesFunctions.LTCodeClass import LTCode

if __name__ == '__main__':
    from CodesFunctions.graphs import *
    import matplotlib.pyplot as plt
    import networkx as nx
    import time


    ########## Crazy-graph encoding
    nrows = 4
    nlayers = 4
    encode_graph = gen_crazy_graph(nrows, nlayers)
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

    # start_time = time.time()
    # tele_meas_full = mycode.full_search_valid_teleportation_meas(trivial_stab_test=False, exclude_input_ys=True)
    # end_time = time.time()
    # print("Time passed for full algorithm:", end_time-start_time, "s")
    # tele_meas_full_weights = list(map(mycode.meas_weight, tele_meas_full))

    start_time = time.time()
    alpha = 2
    tele_meas_SPalg = mycode.SPalgorithm_valid_teleportation_meas(max_m_increase=alpha, test_inouts=True,
                                                                  exclude_input_ys=True, return_evolution=False)
    print(tele_meas_SPalg)

    end_time = time.time()
    print("Time passed for SP algorithm with alpha", alpha, ":", end_time-start_time, "s")
    tele_meas_SPalg_weights = list(map(mycode.meas_weight, tele_meas_SPalg))

    start_time = time.time()
    alpha1 = 3
    tele_meas_SPalg1 = mycode.SPalgorithm_valid_teleportation_meas(max_m_increase=alpha1, test_inouts=True,
                                                                   exclude_input_ys=True, return_evolution=False)
    end_time = time.time()
    print("Time passed for SP algorithm with alpha", alpha, ":", end_time-start_time, "s")
    tele_meas_SPalg_weights1 = list(map(mycode.meas_weight, tele_meas_SPalg1))

    bins = np.arange(mycode.res_graph_num_nodes)
    # plt.hist(tele_meas_full_weights, bins, alpha=0.5, label='Full alg.')
    plt.hist(tele_meas_SPalg_weights, bins, alpha=0.5, label='SP alg.; alpha='+str(alpha))
    plt.hist(tele_meas_SPalg_weights1, bins, alpha=0.5, label='SP alg.; alpha='+str(alpha1))
    plt.legend(loc='upper right')
    plt.show()


    plt.figure()
    mycode.image(with_labels=True)
    plt.show()