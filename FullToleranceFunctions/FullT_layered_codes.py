from FullToleranceFunctions.FullT_analyticalDecoderProbs_treesearch import code_probs_from_decoder_output

###############################################
####### FUNCTIONS FOR CASCADED CODES ##########
###############################################

def casc_codeprobs_X(t, err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z, code_lookup_zwithdir):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list) or isinstance(err_rates_XYZ, tuple):
            return 1, err_rates_XYZ[0]
        else:
            return 1, err_rates_XYZ
    else:
        # TODO: do teleportation cases as well
        t_out, err_rates_out = 0, 0
        t_xi, err_rate_xi = casc_codeprobs_X(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        t_yi, err_rate_yi = casc_codeprobs_Y(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        t_zi, err_rate_zi = casc_codeprobs_Z(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        if isinstance(err_rates_XYZ, list):
            [e_x, e_y, _] = err_rates_XYZ
        else:
            e_x = e_y = err_rates_XYZ
        if isinstance(code_lookup_x, dict):
            temp_t_x, temp_p_x = code_probs_from_decoder_output(code_lookup_x, t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
            temp_t_y, temp_p_y = code_probs_from_decoder_output(code_lookup_y, t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
        else:
            temp_t_x, temp_p_x = code_probs_from_decoder_output(code_lookup_x[layer_ix - 1], t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
            temp_t_y, temp_p_y = code_probs_from_decoder_output(code_lookup_y[layer_ix - 1], t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
        if layer_ix == 0:
            return temp_t_x, temp_p_x
        else:
            p_x = e_x + temp_p_x - e_x * temp_p_x
            p_y = e_y + temp_p_y - e_y * temp_p_y
            # TODO: The choice of whether to do X or Y here depends only on error rate and not loss, finding a good way to use both might improve things (they seem to be quite related though)
            if p_y < p_x:
                return temp_t_y, p_y
            else:
                return temp_t_x, p_x

def casc_codeprobs_Y(t, err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z, code_lookup_zwithdir):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list) or isinstance(err_rates_XYZ, tuple):
            return 1, err_rates_XYZ[1]
        else:
            return 1, err_rates_XYZ
    else:
        # TODO: do teleportation cases as well
        t_out, err_rates_out = 0, 0
        t_xi, err_rate_xi = casc_codeprobs_X(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        t_yi, err_rate_yi = casc_codeprobs_Y(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        t_zi, err_rate_zi = casc_codeprobs_Z(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        if isinstance(err_rates_XYZ, list):
            [e_x, e_y, _] = err_rates_XYZ
        else:
            e_x = e_y = err_rates_XYZ
        if isinstance(code_lookup_x, dict):
            temp_t_x, temp_p_x = code_probs_from_decoder_output(code_lookup_x, t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
            temp_t_y, temp_p_y = code_probs_from_decoder_output(code_lookup_y, t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
        else:
            temp_t_x, temp_p_x = code_probs_from_decoder_output(code_lookup_x[layer_ix - 1], t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
            temp_t_y, temp_p_y = code_probs_from_decoder_output(code_lookup_y[layer_ix - 1], t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
        if layer_ix == 0:
            return temp_t_x, temp_p_x
        else:
            p_x = e_x + temp_p_x - e_x * temp_p_x
            p_y = e_y + temp_p_y - e_y * temp_p_y
            # TODO: The choice of whether to do X or Y here depends only on error rate and not loss, finding a good way to use both might improve things (they seem to be quite related though)
            if p_y < p_x:
                return temp_t_y, p_y
            else:
                return temp_t_x, p_x


def casc_codeprobs_Z(t, err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z, code_lookup_zwithdir):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list):
            return 0, err_rates_XYZ[2]
        else:
            return 0, err_rates_XYZ
    else:
        # TODO: do teleportation cases as well
        t_out, err_rates_out = 0, 0
        t_xi, err_rate_xi = casc_codeprobs_X(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        t_yi, err_rate_yi = casc_codeprobs_Y(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)
        t_zi, err_rate_zi = casc_codeprobs_Z(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z, code_lookup_zwithdir)

        if isinstance(code_lookup_z, dict):
            t_z_somemeas, temp_p_zind = code_probs_from_decoder_output(code_lookup_z, t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
            _, temp_p_zdirind = code_probs_from_decoder_output(code_lookup_zwithdir, t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
        else:
            t_z_somemeas, temp_p_zind = code_probs_from_decoder_output(code_lookup_z[layer_ix - 1], t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
            _, temp_p_zdirind = code_probs_from_decoder_output(code_lookup_zwithdir[layer_ix - 1], t, err_rate_xi, err_prob_Y=err_rate_yi, err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)

        if layer_ix == 0:
            return t_z_somemeas, temp_p_zind
        else:
            ### calculate the effective transmission and error rate considering that the cases with direct measurement,
            ### indirect measurement, and both are all good for loss but provide significantly different error rates.
            temp_t = t_z_somemeas * (1 - t)  # indirect measurement only
            temp_t += t * (1 - t_z_somemeas)  # direct measurement only
            temp_t += t_z_somemeas * t  # both direct and indirect measurements

            temp_p = t_z_somemeas * (1 - t) * temp_p_zind  # indirect measurement only
            if isinstance(err_rates_XYZ, list):
                temp_p += t * (1 - t_z_somemeas) * err_rates_XYZ[2]  # direct measurement only
            else:
                temp_p += t * (1 - t_z_somemeas) * err_rates_XYZ  # direct measurement only
            temp_p += t_z_somemeas * t * temp_p_zdirind  # both direct and indirect measurements
            temp_p = temp_p / temp_t  # renormalize probability, conditioning on having a successful measurement.

            return temp_t, temp_p

###################################################
####### FUNCTIONS FOR CONCATENATED CODES ##########
###################################################


def conc_codeprobs_X(t, err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list) or isinstance(err_rates_XYZ, tuple):
            return 1, err_rates_XYZ[0]
        else:
            return 1, err_rates_XYZ
    else:
        if layer_ix == N_layers - 1:
            t_dir = t
        else:
            t_dir = 1
        # TODO: do teleportation cases as well
        t_out, err_rates_out = 0, 0
        t_xi, err_rate_xi = conc_codeprobs_X(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
        t_yi, err_rate_yi = conc_codeprobs_Y(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
        t_zi, err_rate_zi = conc_codeprobs_Z(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
    if isinstance(code_lookup_x, dict):
        return code_probs_from_decoder_output(code_lookup_x, t_dir, err_rate_xi, err_prob_Y=err_rate_yi,
                                              err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
    else:
        return code_probs_from_decoder_output(code_lookup_x[layer_ix - 1], t_dir, err_rate_xi, err_prob_Y=err_rate_yi,
                                              err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)


def conc_codeprobs_Y(t, err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    # print('Doing layer_ix', layer_ix)
    if layer_ix == N_layers:
        # print('returning', err_rates_XYZ)
        if isinstance(err_rates_XYZ, list) or isinstance(err_rates_XYZ, tuple):
            return 1, err_rates_XYZ[1]
        else:
            return 1, err_rates_XYZ
    else:
        if layer_ix == N_layers - 1:
            t_dir = t
        else:
            t_dir = 1
        # TODO: do teleportation cases as well
        t_out, err_rates_out = 0, 0
        t_xi, err_rate_xi = conc_codeprobs_X(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
        t_yi, err_rate_yi = conc_codeprobs_Y(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
        t_zi, err_rate_zi = conc_codeprobs_Z(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
        # print('layer', layer_ix, 'uses:', t_xi, err_rate_xi, t_yi, err_rate_yi, t_zi, err_rate_zi)
    if isinstance(code_lookup_x, dict):
        # print('Using2:', t_xi, err_rate_xi, t_yi, err_rate_yi, t_zi, err_rate_zi)
        return code_probs_from_decoder_output(code_lookup_y, t_dir, err_rate_xi, err_prob_Y=err_rate_yi,
                                              err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
    else:
        return code_probs_from_decoder_output(code_lookup_y[layer_ix - 1], t_dir, err_rate_xi, err_prob_Y=err_rate_yi,
                                              err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)


def conc_codeprobs_Z(t, err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list) or isinstance(err_rates_XYZ, tuple):
            return 0, err_rates_XYZ[1]
        else:
            return 0, err_rates_XYZ
    else:
        if layer_ix == N_layers - 1:
            t_dir = t
        else:
            t_dir = 1
        # TODO: do teleportation cases as well
        t_out, err_rates_out = 0, 0
        t_xi, err_rate_xi = conc_codeprobs_X(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
        t_yi, err_rate_yi = conc_codeprobs_Y(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
        t_zi, err_rate_zi = conc_codeprobs_Z(t, err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                             code_lookup_z)
    if isinstance(code_lookup_x, dict):
        return code_probs_from_decoder_output(code_lookup_z, t_dir, err_rate_xi, err_prob_Y=err_rate_yi,
                                              err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)
    else:
        return code_probs_from_decoder_output(code_lookup_z[layer_ix - 1], t_dir, err_rate_xi, err_prob_Y=err_rate_yi,
                                              err_prob_Z=err_rate_zi, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi, t_out=t_out)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx
    from FullToleranceFunctions.FullT_Decoders_Classes import FullT_IndMeasDecoder
    from FullToleranceFunctions.FullT_analyticalDecoderProbs_treesearch import get_FullTdecoder_succpob_treesearch

    from LossToleranceFunctions.LT_Decoders_Classes import LT_FullDecoder, LT_IndMeasDecoder
    from LossToleranceFunctions.LT_analyticalDecodersProbs_treesearch import get_LTdecoder_succpob_treesearch
    from LossToleranceFunctions.LT_layered_codes import conc_prob_X, conc_prob_Y, conc_prob_Z, conc_prob_full

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    ## three graph
    branching = [5]
    graph = gen_tree_graph(branching)
    gstate = GraphState(graph)

    # ### fully connected graph
    # graph = gen_fullyconnected_graph(7)
    # gstate = GraphState(graph)

    # # ring graph
    # graph = gen_ring_graph(5)
    # gstate = GraphState(graph)

    # ### Generate random graph
    # graph = nx.Graph()
    # graph.add_nodes_from(range(2))
    # while not nx.is_connected(graph):
    #     graph = gen_random_connected_graph(6)
    # gstate = GraphState(graph)

    #################################################################
    ######################## RUN DECODERS ###########################
    #################################################################

    # get expression for full decoders
    FullT_decoder_X = FullT_IndMeasDecoder(gstate, 'X', in_qubit)
    FullT_decoder_Y = FullT_IndMeasDecoder(gstate, 'Y', in_qubit)
    FullT_decoder_Z = FullT_IndMeasDecoder(gstate, 'Z', in_qubit)

    decoder_output_X = get_FullTdecoder_succpob_treesearch(FullT_decoder_X)
    decoder_output_Y = get_FullTdecoder_succpob_treesearch(FullT_decoder_Y)
    decoder_output_Z = get_FullTdecoder_succpob_treesearch(FullT_decoder_Z)
    decoder_output_Z_withinput = get_FullTdecoder_succpob_treesearch(FullT_decoder_Z, include_direct_meas=True)

    ############### Plots
    transmission = 0.9
    # transmission = 0.9639

    gstate.image(input_qubits=[in_qubit])
    plt.show()

    ind_meas_pauli = 'Z'
    err_rate_list = np.linspace(0, 0.3, 15)
    # err_rate_list = np.linspace(0, 0.3, 1)

    num_layers_list = [1, 2, 3, 4]
    # num_layers_list = [1, 2]

    if ind_meas_pauli == 'X':
        conc_errorrate_func = conc_codeprobs_X
        casc_errorrate_func = casc_codeprobs_X
    elif ind_meas_pauli == 'Y':
        conc_errorrate_func = conc_codeprobs_Y
        casc_errorrate_func = casc_codeprobs_Y
    else:
        conc_errorrate_func = conc_codeprobs_Z
        casc_errorrate_func = casc_codeprobs_Z

    plt.plot(err_rate_list, err_rate_list, 'k:', label='Direct')

    ###### TEST CASCADED CODES
    for N_layers in num_layers_list:
        this_code_results_list = [
            casc_errorrate_func(transmission, noise_rate, 0, N_layers, decoder_output_X, decoder_output_Y, decoder_output_Z, decoder_output_Z_withinput)
            for noise_rate in err_rate_list]
        this_code_error_rates_list = [x[1] for x in this_code_results_list]
        plt.plot(err_rate_list, this_code_error_rates_list, linestyle='--', label=str(N_layers) + r' Casc.$t_{eff}$:'+ str(this_code_results_list[0][0]))

    ###### TEST CONCATENATED CODES
    for N_layers in num_layers_list:
        # print('\n\n')
        # print('N_layers', N_layers)
        this_code_results_list = [
            conc_errorrate_func(transmission, noise_rate, 0, N_layers, decoder_output_X, decoder_output_Y, decoder_output_Z)
            for noise_rate in err_rate_list]
        # print(this_code_results_list)
        this_code_error_rates_list = [x[1] for x in this_code_results_list]
        plt.plot(err_rate_list, this_code_error_rates_list, label=str(N_layers)+r' Conc., $t_{eff}$:'+ str(this_code_results_list[0][0]))

    # #### Check consistency with loss-tolerance alone.
    # # get expression for full decoder
    # decoder = LT_FullDecoder(gstate, in_qubit)
    # code_prob_expr_full = get_LTdecoder_succpob_treesearch(decoder)
    # # get expression for ind X measurement
    # decoder = LT_IndMeasDecoder(gstate, 'X', in_qubit)
    # code_prob_expr_x = get_LTdecoder_succpob_treesearch(decoder)
    # # get expression for ind Y measurement
    # decoder = LT_IndMeasDecoder(gstate, 'Y', in_qubit)
    # code_prob_expr_y = get_LTdecoder_succpob_treesearch(decoder)
    # # get expression for ind Z measurement
    # decoder = LT_IndMeasDecoder(gstate, 'Z', in_qubit)
    # code_prob_expr_z = get_LTdecoder_succpob_treesearch(decoder)
    # print('\n\ntheo values from LT:')
    # for N_layers in num_layers_list:
    #     prob = conc_prob_Y(transmission, 0, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
    #     print('N_layers:', N_layers, 'prob:', prob)

    plt.xlabel('Physical error probability')
    plt.ylabel('Logical error probability ' + ind_meas_pauli + ' meas.')
    plt.title('Transmission:'+str(transmission))
    plt.legend()
    plt.show()
