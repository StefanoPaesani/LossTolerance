from ErrorCorrectionFunctions.EC_DecoderClasses import log_op_error_prob_from_lookup_dict


#### CASCADED CODES


def casc_errorrate_X(err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z,
                     code_lookup_z_withinput):
    if layer_ix == N_layers+1:
        return 0
    else:
        err_rate_xi = casc_errorrate_X(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        err_rate_yi = casc_errorrate_Y(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        err_rate_zi = casc_errorrate_Z(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        if isinstance(err_rates_XYZ, list):
            [e_x, e_y, _] = err_rates_XYZ
        else:
            e_x = e_y = err_rates_XYZ
        if isinstance(code_lookup_x, dict):
            temp_p_x = log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
            temp_p_y = log_op_error_prob_from_lookup_dict(code_lookup_y, err_rate_xi, err_rate_yi, err_rate_zi)
        else:
            temp_p_x = log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi,
                                                          err_rate_zi)
            temp_p_y = log_op_error_prob_from_lookup_dict(code_lookup_y[layer_ix - 1], err_rate_xi, err_rate_yi,
                                                          err_rate_zi)
        if layer_ix == 0:
            return temp_p_x
        else:
            p_x = e_x * (1 - temp_p_x) + temp_p_x * (1 - e_x)
            p_y = e_y * (1 - temp_p_y) + temp_p_y * (1 - e_y)
            if layer_ix < N_layers:
                return min(p_x, p_y)
            else:
                return p_x


def casc_errorrate_Y(err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z,
                     code_lookup_z_withinput):
    if layer_ix == N_layers+1:
        return 0
    else:
        err_rate_xi = casc_errorrate_X(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        err_rate_yi = casc_errorrate_Y(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        err_rate_zi = casc_errorrate_Z(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        if isinstance(err_rates_XYZ, list):
            [e_x, e_y, _] = err_rates_XYZ
        else:
            e_x = e_y = err_rates_XYZ
        if isinstance(code_lookup_x, dict):
            temp_p_x = log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
            temp_p_y = log_op_error_prob_from_lookup_dict(code_lookup_y, err_rate_xi, err_rate_yi, err_rate_zi)

        else:
            temp_p_x = log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi,
                                                          err_rate_zi)
            temp_p_y = log_op_error_prob_from_lookup_dict(code_lookup_y[layer_ix - 1], err_rate_xi, err_rate_yi,
                                                          err_rate_zi)
        p_x = e_x * (1 - temp_p_x) + temp_p_x * (1 - e_x)
        p_y = e_y * (1 - temp_p_y) + temp_p_y * (1 - e_y)
        if layer_ix == 0:
            return temp_p_x
        else:
            p_x = e_x * (1 - temp_p_x) + temp_p_x * (1 - e_x)
            p_y = e_y * (1 - temp_p_y) + temp_p_y * (1 - e_y)
            if layer_ix < N_layers:
                return min(p_x, p_y)
            else:
                return p_x


def casc_errorrate_Z(err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z,
                     code_lookup_z_withinput):
    if layer_ix == N_layers + 1:
        return 0.
    elif layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list):
            return err_rates_XYZ[2]
        else:
            return err_rates_XYZ
    else:
        err_rate_xi = casc_errorrate_X(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        err_rate_yi = casc_errorrate_Y(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)
        err_rate_zi = casc_errorrate_Z(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z, code_lookup_z_withinput)

        if isinstance(code_lookup_z, dict):
            if layer_ix == 0:
                return log_op_error_prob_from_lookup_dict(code_lookup_z, err_rate_xi, err_rate_yi, err_rate_zi)
            else:
                return log_op_error_prob_from_lookup_dict(code_lookup_z_withinput, err_rate_xi, err_rate_yi, err_rate_zi)
        else:

            if layer_ix == 0:
                return log_op_error_prob_from_lookup_dict(code_lookup_z[layer_ix - 1], err_rate_xi,
                                                          err_rate_yi, err_rate_zi)
            else:
                return log_op_error_prob_from_lookup_dict(code_lookup_z_withinput[layer_ix - 1], err_rate_xi,
                                                          err_rate_yi, err_rate_zi)


#### CONCATENATED CODES

def conc_errorrate_X(err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list):
            return err_rates_XYZ[0]
        else:
            return err_rates_XYZ
    else:
        err_rate_xi = conc_errorrate_X(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        err_rate_yi = conc_errorrate_Y(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        err_rate_zi = conc_errorrate_Z(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        if isinstance(code_lookup_x, dict):
            return log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
        else:
            return log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi,
                                                      err_rate_zi)


def conc_errorrate_Y(err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list):
            return err_rates_XYZ[0]
        else:
            return err_rates_XYZ
    else:
        err_rate_xi = conc_errorrate_X(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        err_rate_yi = conc_errorrate_Y(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        err_rate_zi = conc_errorrate_Z(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        if isinstance(code_lookup_x, dict):
            return log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
        else:
            return log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi,
                                                      err_rate_zi)


def conc_errorrate_Z(err_rates_XYZ, layer_ix, N_layers, code_lookup_x, code_lookup_y, code_lookup_z):
    if layer_ix == N_layers:
        if isinstance(err_rates_XYZ, list):
            return err_rates_XYZ[0]
        else:
            return err_rates_XYZ
    else:
        err_rate_xi = conc_errorrate_X(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        err_rate_yi = conc_errorrate_Y(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        err_rate_zi = conc_errorrate_Z(err_rates_XYZ, layer_ix + 1, N_layers, code_lookup_x, code_lookup_y,
                                       code_lookup_z)
        if isinstance(code_lookup_x, dict):
            return log_op_error_prob_from_lookup_dict(code_lookup_x, err_rate_xi, err_rate_yi, err_rate_zi)
        else:
            return log_op_error_prob_from_lookup_dict(code_lookup_x[layer_ix - 1], err_rate_xi, err_rate_yi,
                                                      err_rate_zi)


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
    syndromes_probs_dict_Z_withinput = ind_meas_EC_decoder(gstate, 'Z', in_qubit, include_direct_meas=True,
                                                           max_error_num=None)

    ############### Plots
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    ind_meas_pauli = 'Y'
    err_rate_list = np.linspace(0, 0.3, 15)

    num_layers_list = [1, 2, 3, 4]

    if ind_meas_pauli == 'X':
        conc_errorrate_func = conc_errorrate_X
        casc_errorrate_func = casc_errorrate_X
    elif ind_meas_pauli == 'Y':
        conc_errorrate_func = conc_errorrate_Y
        casc_errorrate_func = casc_errorrate_Y
    else:
        conc_errorrate_func = conc_errorrate_Z
        casc_errorrate_func = casc_errorrate_Z

    plt.plot(err_rate_list, err_rate_list, 'k:', label='Direct')

    ###### TEST CASCADED CODES
    for N_layers in num_layers_list:
        this_code_prob_list = [
            casc_errorrate_func(t, 0, N_layers, syndromes_probs_dict_X, syndromes_probs_dict_Y, syndromes_probs_dict_Z,
                                syndromes_probs_dict_Z_withinput)
            for t in err_rate_list]
        plt.plot(err_rate_list, this_code_prob_list, linestyle='--', label=str(N_layers) + ' Casc.')

    ###### TEST CONCATENATED CODES
    for N_layers in num_layers_list:
        this_code_prob_list = [
            conc_errorrate_func(t, 0, N_layers, syndromes_probs_dict_X, syndromes_probs_dict_Y, syndromes_probs_dict_Z)
            for t in err_rate_list]
        plt.plot(err_rate_list, this_code_prob_list, label=str(N_layers)+' Conc.')

    plt.xlabel('Physical error probability')
    plt.ylabel('Logical error probability ' + ind_meas_pauli + ' meas.')
    plt.legend()
    plt.show()
