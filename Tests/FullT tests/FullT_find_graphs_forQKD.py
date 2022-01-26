if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx
    import numpy as np

    from FullToleranceFunctions.FullT_Decoders_Classes import FullT_IndMeasDecoder
    from FullToleranceFunctions.FullT_analyticalDecoderProbs_treesearch import get_FullTdecoder_succpob_treesearch, \
        code_probs_from_decoder_output



    #################################################################################
    ######################### TESTS OVER MANY GRAPHS ################################
    #################################################################################

    in_qubit = 0
    single_test_noise = 0.05
    num_qubits = 9

    noise_levels_plots = np.linspace(0, 0.3, 15)

    num_max_graphs = 1000
    printing_rate = 1  # 10

    measurements_required = ['Z', 'X'] # found none: ['X', 'Y'],  # ['Tele', 'X', Y', 'Z']

    for graph_ix in range(num_max_graphs):
        if graph_ix % printing_rate == 0:
            print('Graph', graph_ix, 'of', num_max_graphs)
        graph = gen_random_connected_graph(num_qubits)

        if nx.is_connected(graph):
            gstate = GraphState(graph)
        ###### Run decoders
            this_code_results = [code_probs_from_decoder_output(get_FullTdecoder_succpob_treesearch(FullT_IndMeasDecoder(gstate, measurement, in_qubit)), 1-single_test_noise, single_test_noise) for measurement in measurements_required]
            succ_prob_rates = [(1-these_results[0]) < single_test_noise for these_results in this_code_results]
            error_rates = [these_results[1] < single_test_noise for these_results in this_code_results]
            if (np.all(succ_prob_rates)) and (np.all(error_rates)):
                print('Found graph which is both LT and EC for this problem!!!')
                print(this_code_results)
                gstate.image(input_qubits=[in_qubit])
                plt.show()

                # loss_probs = [1 - probsucc_poly_fromexpress(1 - loss, code_LT_prob_expr) for loss in noise_levels_plots]
                # error_rates = [EC_prob_func(EC_decoder_output, err_prob) for err_prob in noise_levels_plots]
                #
                # plt.plot(noise_levels_plots, noise_levels_plots, 'k:', label='')
                # plt.plot(noise_levels_plots, loss_probs, 'r', label='LT decoder', linewidth=2)
                # plt.plot(noise_levels_plots, error_rates, 'b', label='EC decoder')
                # plt.xlabel('Physical error probability')
                # plt.ylabel('Logical error probability')
                # plt.legend()
                # plt.show()
        else:
            print("Graph was not connected")