from ErrorCorrectionFunctions.EC_DecoderClasses import log_op_error_prob_from_lookup_dict

def conc_errorrate_X(err_rate, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        return err_rate
    else:
        err_rate_xi = conc_errorrate_X(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        err_rate_yi = conc_errorrate_Y(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        err_rate_zi = conc_errorrate_Z(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        if isinstance(code_lookup_x, dict):
            # return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x)
            return log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
        else:
            # return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x[layer_ix - 1])
            return log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi, err_rate_zi)


def conc_errorrate_Y(err_rate, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        return err_rate
    else:
        err_rate_xi = conc_errorrate_X(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        err_rate_yi = conc_errorrate_Y(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        err_rate_zi = conc_errorrate_Z(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        if isinstance(code_lookup_x, dict):
            # return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x)
            return log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
        else:
            # return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x[layer_ix - 1])
            return log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi, err_rate_zi)


def conc_errorrate_Z(err_rate, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        return err_rate
    else:
        err_rate_xi = conc_errorrate_X(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        err_rate_yi = conc_errorrate_Y(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        err_rate_zi = conc_errorrate_Z(err_rate, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y, code_lookup_z)
        if isinstance(code_lookup_x, dict):
            # return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x)
            return log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
        else:
            # return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x[layer_ix - 1])
            return log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi, err_rate_zi)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx

    from ErrorCorrectionFunctions.EC_DecoderClasses import ind_meas_EC_decoder

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    # ## three graph
    # branching = [2]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    # ### fully connected graph
    # graph = gen_fullyconnected_graph(7)
    # gstate = GraphState(graph)

    ## ring graph
    # graph = gen_ring_graph(5)
    # gstate = GraphState(graph)

    ### Generate random graph
    graph = gen_random_connected_graph(6)
    gstate = GraphState(graph)

    #################################################################
    ######################## RUN DECODERS ###########################
    #################################################################

    # get expression for full decoder
    syndromes_probs_dict_X = ind_meas_EC_decoder(gstate, 'X', in_qubit, max_error_num=None)
    syndromes_probs_dict_Y = ind_meas_EC_decoder(gstate, 'Y', in_qubit, max_error_num=None)
    syndromes_probs_dict_Z = ind_meas_EC_decoder(gstate, 'Z', in_qubit, max_error_num=None)


    #################################################################
    ################ TESTS CONCATENATED CODES #######################
    #################################################################

    ############### Plots
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    err_rate_list = np.linspace(0, 0.1, 10)

    num_layers_list = [0, 1, 2, 3]

    plt.plot(err_rate_list, err_rate_list, 'k:', label='Direct')
    for N_layers in num_layers_list:
        this_code_prob_list = [
            conc_errorrate_Y(t, 0, N_layers, syndromes_probs_dict_X, syndromes_probs_dict_Y, syndromes_probs_dict_Z)
            for t in err_rate_list]
        plt.plot(err_rate_list, this_code_prob_list, label=str(N_layers))
    plt.xlabel('Physical error probability')
    plt.ylabel('Logical error probability')
    plt.legend()
    plt.show()
