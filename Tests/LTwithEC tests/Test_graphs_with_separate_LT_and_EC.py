if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx
    import numpy as np

    from LossToleranceFunctions.LT_Decoders_Classes import LT_FullDecoder, LT_IndMeasDecoder
    from LossToleranceFunctions.LT_analyticalDecodersProbs_treesearch import get_LTdecoder_succpob_treesearch, \
        probsucc_poly_fromexpress
    from ErrorCorrectionFunctions.EC_DecoderClasses import ind_meas_EC_decoder, log_op_error_prob_from_lookup_dict, \
        teleportation_EC_decoder, teleport_error_prob_from_lookup_dict

    #################################################################
    ###################### SINGLE GRAPHS TEST #######################
    #################################################################

    # ######## Define type of problem
    #
    # in_qubit = 0
    # noise_levels = np.linspace(0, 0.3, 15)
    #
    # measurement = 'Tele'  # ['Tele', 'X', Y', 'Z']
    #
    # ######## Define graph state
    #
    # # ## three graph
    # # branching = [2]
    # # graph = gen_tree_graph(branching)
    # # gstate = GraphState(graph)
    #
    # # ### fully connected graph
    # # graph = gen_fullyconnected_graph(7)
    # # gstate = GraphState(graph)
    #
    # # ### ring graph
    # # graph = gen_ring_graph(5)
    # # gstate = GraphState(graph)
    #
    # ### Generate random graph
    # graph = gen_random_connected_graph(6)
    # gstate = GraphState(graph)
    #
    # ############
    # gstate.image(input_qubits=[in_qubit])
    # plt.show()
    #
    # ###### Run LT decoder
    # if measurement in ['X', 'Y', 'Z']:
    #     LT_decoder = LT_IndMeasDecoder(gstate, measurement, in_qubit)
    # else:
    #     LT_decoder = LT_FullDecoder(gstate, in_qubit)
    # # get loss probabilities for LT decoder
    # code_LT_prob_expr = get_LTdecoder_succpob_treesearch(LT_decoder)
    # loss_probs = [1 - probsucc_poly_fromexpress(1 - loss, code_LT_prob_expr) for loss in noise_levels]
    #
    # ###### Run EC decoder
    # if measurement in ['X', 'Y', 'Z']:
    #     EC_decoder_output = ind_meas_EC_decoder(gstate, measurement, in_qubit)
    #     EC_prob_func = log_op_error_prob_from_lookup_dict
    # else:
    #     EC_decoder_output = teleportation_EC_decoder(gstate, in_qubit)
    #     EC_prob_func = teleport_error_prob_from_lookup_dict
    # # get logical error probabilities for EC decoder
    # error_rates = [EC_prob_func(EC_decoder_output, err_prob) for err_prob in noise_levels]
    #
    # ##################################### Plots
    #
    # plt.plot(noise_levels, noise_levels, 'k:', label='', )
    # plt.plot(noise_levels, loss_probs, 'r', label='LT decoder', linewidth=2)
    # plt.plot(noise_levels, error_rates, 'b', label='EC decoder')
    # plt.xlabel('Physical error probability')
    # plt.ylabel('Logical error probability')
    # plt.legend()
    # plt.show()

    #################################################################################
    ######################### TESTS OVER MANY GRAPHS ################################
    #################################################################################

    in_qubit = 0
    single_test_noise = 0.05
    num_qubits = 8

    noise_levels_plots = np.linspace(0, 0.3, 15)

    num_max_graphs = 1000
    printing_rate = 1  # 10

    measurement = 'Tele'  # ['Tele', 'X', Y', 'Z']

    for graph_ix in range(num_max_graphs):
        if graph_ix % printing_rate == 0:
            print('Graph', graph_ix, 'of', num_max_graphs)
        graph = gen_random_connected_graph(num_qubits)
        gstate = GraphState(graph)

        ###### Run LT decoder
        if measurement in ['X', 'Y', 'Z']:
            LT_decoder = LT_IndMeasDecoder(gstate, measurement, in_qubit)
        else:
            LT_decoder = LT_FullDecoder(gstate, in_qubit)
        # get loss probabilities for LT decoder
        code_LT_prob_expr = get_LTdecoder_succpob_treesearch(LT_decoder)
        this_loss = 1 - probsucc_poly_fromexpress(1 - single_test_noise, code_LT_prob_expr)

        ###### Run EC decoder
        if measurement in ['X', 'Y', 'Z']:
            EC_decoder_output = ind_meas_EC_decoder(gstate, measurement, in_qubit)
            EC_prob_func = log_op_error_prob_from_lookup_dict
        else:
            EC_decoder_output = teleportation_EC_decoder(gstate, in_qubit)
            EC_prob_func = teleport_error_prob_from_lookup_dict
        # get logical error probabilities for EC decoder
        this_error_rate = EC_prob_func(EC_decoder_output, single_test_noise)

        if this_error_rate < single_test_noise and this_loss < single_test_noise:
            print('Found graph which is both LT and EC for this problem!!!')
            gstate.image(input_qubits=[in_qubit])
            plt.show()

            loss_probs = [1 - probsucc_poly_fromexpress(1 - loss, code_LT_prob_expr) for loss in noise_levels_plots]
            error_rates = [EC_prob_func(EC_decoder_output, err_prob) for err_prob in noise_levels_plots]

            plt.plot(noise_levels_plots, noise_levels_plots, 'k:', label='')
            plt.plot(noise_levels_plots, loss_probs, 'r', label='LT decoder', linewidth=2)
            plt.plot(noise_levels_plots, error_rates, 'b', label='EC decoder')
            plt.xlabel('Physical error probability')
            plt.ylabel('Logical error probability')
            plt.legend()
            plt.show()
