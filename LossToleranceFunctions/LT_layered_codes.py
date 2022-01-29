###############################################
####### FUNCTIONS FOR CASCADED CODES ##########
###############################################

# Function that converts the polynomial expression into the success probability as a function of:
# transmission t, and indirect measurement probabilities p_xyi, p_zi
# The terms in the expression are in the order: (OUT_OUT, OUT_Z, OUT_na, X_X, X_Z, X_na, Y_Y, Y_Z, Y_na, Z_Z, Z_na)
def probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, poly_express):
    return sum(
        [poly_express[term] *
         ((t*t_xyi)**term[0]) *
         (((1-t)*t_zi)**term[1]) *
         (((1-t)*(1-t_zi) + t*(1-t_xyi))**term[2]) *

         ((t * t_xyi) ** term[3]) *
         (((1 - t) * t_zi) ** term[4]) *
         (((1 - t) * (1 - t_zi) + t * (1 - t_xyi)) ** term[5]) *

         ((t * t_xyi) ** term[6]) *
         (((1 - t) * t_zi) ** term[7]) *
         (((1 - t) * (1 - t_zi) + t * (1 - t_xyi)) ** term[8]) *

         ((t+(1-t)*t_zi)**term[9]) *
         ((1-t-(1-t)*t_zi)**term[10])
         for term in poly_express])


def cascade_prob_X(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return 1
    else:
        p_x = cascade_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        if isinstance(code_prob_expr_x, dict):
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_x)
        else:
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_x[layer_ix - 1])


def cascade_prob_Y(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return 1
    else:
        p_x = cascade_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        if isinstance(code_prob_expr_y, dict):
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_y)
        else:
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_y[layer_ix - 1])


def cascade_prob_Z(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return 0
    else:
        p_x = cascade_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        if isinstance(code_prob_expr_z, dict):
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_z)
        else:
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_z[layer_ix - 1])


def cascade_prob_full(t, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z, N_layers=None):
    if N_layers is None:
        if isinstance(code_prob_expr_full, dict):
            raise ValueError("N_layers needs to be specified if single code is provided")
        else:
            N_layers = len(code_prob_expr_full)

    if N_layers == 0:
        return t
    else:
        p_x = cascade_prob_X(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        if isinstance(code_prob_expr_full, dict):
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_full)
        else:
            return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_full[0])


###################################################
####### FUNCTIONS FOR CONCATENATED CODES ##########
###################################################

def probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, poly_express):
    return sum(
        [poly_express[term] * (0 if any([term[1] > 0, term[4] > 0, term[7] > 0]) else 1) *

         (t_out ** term[0]) *
         ((1 - t_out) ** term[2]) *

         (t_xi ** term[3]) *
         ((1 - t_xi) ** term[5]) *

         (t_yi ** term[6]) *
         ((1 - t_yi) ** term[8]) *

         (t_zi ** term[9]) *
         ((1 - t_zi) ** term[10])
         for term in poly_express])


def conc_prob_out(t, layer_ix, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return t
    else:
        t_out = conc_prob_out(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                              code_prob_expr_z)
        t_xi = conc_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_yi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_zi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        if isinstance(code_prob_expr_full, dict):
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_full)
        else:
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_full[layer_ix - 1])


def conc_prob_X(t, layer_ix, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return t
    else:
        t_out = conc_prob_out(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                              code_prob_expr_z)
        t_xi = conc_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_yi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_zi = conc_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        if isinstance(code_prob_expr_x, dict):
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x)
        else:
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_x[layer_ix - 1])


def conc_prob_Y(t, layer_ix, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return t
    else:
        t_out = conc_prob_out(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                              code_prob_expr_z)
        t_xi = conc_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_yi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_zi = conc_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        if isinstance(code_prob_expr_y, dict):
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_y)
        else:
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_y[layer_ix - 1])


def conc_prob_Z(t, layer_ix, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return t
    else:
        t_out = conc_prob_out(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                              code_prob_expr_z)
        t_xi = conc_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_yi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        t_zi = conc_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y,
                           code_prob_expr_z)
        if isinstance(code_prob_expr_z, dict):
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_z)
        else:
            return probsucc_poly_fromexpress_conc(t_out, t_xi, t_yi, t_zi, code_prob_expr_z[layer_ix - 1])


def conc_prob_full(t, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z, N_layers=None):
    if N_layers is None:
        if isinstance(code_prob_expr_full, dict):
            raise ValueError("N_layers needs to be specified if single code is provided")
        else:
            N_layers = len(code_prob_expr_full)

    if N_layers == 0:
        return t
    else:
        return conc_prob_out(t, 0, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)


########################################################################################################################
##############################
###          MAIN          ###
##############################

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx

    from LossToleranceFunctions.LT_Decoders_Classes import LT_FullDecoder, LT_IndMeasDecoder
    from LossToleranceFunctions.LT_analyticalDecodersProbs_treesearch import get_LTdecoder_succpob_treesearch

    # from itertools import chain

    branching = None

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    # ## three graph
    # branching = [2]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    ### fully connected graph
    # graph = gen_fullyconnected_graph(7)
    # gstate = GraphState(graph)

    # ## ring graph
    # graph = gen_ring_graph(5)
    # gstate = GraphState(graph)

    ### star graph
    graph = gen_star_graph(4)
    gstate = GraphState(graph)

    #################################################################
    ######################## RUN DECODERS ###########################
    #################################################################

    # get expression for full decoder
    decoder = LT_FullDecoder(gstate, in_qubit)
    code_prob_expr_full = get_LTdecoder_succpob_treesearch(decoder)

    # get expression for ind X measurement
    decoder = LT_IndMeasDecoder(gstate, 'X', in_qubit)
    code_prob_expr_x = get_LTdecoder_succpob_treesearch(decoder)

    # get expression for ind Y measurement
    decoder = LT_IndMeasDecoder(gstate, 'Y', in_qubit)
    code_prob_expr_y = get_LTdecoder_succpob_treesearch(decoder)

    # get expression for ind Z measurement
    decoder = LT_IndMeasDecoder(gstate, 'Z', in_qubit)
    code_prob_expr_z = get_LTdecoder_succpob_treesearch(decoder)

    #################################################################
    ################## TESTS CASCADED CODES #########################
    #################################################################

    # ##### Plots
    # gstate.image(input_qubits=[in_qubit])
    # plt.show()
    #
    # t_list = np.linspace(0, 1, 30)
    #
    # num_layers_list = [1, 2, 3, 4]
    # # num_layers_list = [0, 1, 2, 3, 4, 5]
    # # num_layers_list = [0, 1, 2, 15]
    #
    # plt.plot(t_list, t_list, 'k:', label='Direct')
    # for N_layers in num_layers_list:
    #     this_code_prob_list = [
    #         cascade_prob_full(t, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z, N_layers)
    #         for t in t_list]
    #     plt.plot(t_list, this_code_prob_list, label=str(N_layers))
    #
    # ### code with t_xyi=t_zi=1
    # this_code_prob_list = [
    #     probsucc_poly_fromexpress_casc(t, 1, 1, code_prob_expr_full) for t
    #     in t_list]
    # plt.plot(t_list, this_code_prob_list, 'k--', label='Asymptotic')
    #
    # plt.legend()
    # plt.show()

    #################################################################
    ################ TESTS CONCATENATED CODES #######################
    #################################################################

    # ############### Plots
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    t_list = np.linspace(0, 1, 30)

    num_layers_list = [1, 2, 3, 4]
    # num_layers_list = [0, 1, 2, 15]


    plt.plot(t_list, t_list, 'k:', label='Direct')
    for N_layers in num_layers_list:
        this_code_prob_list = [
            conc_prob_full(t, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z, N_layers)
            for t in t_list]
        # this_code_prob_list = [
        #     conc_prob_X(t, 0, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        #     for t in t_list]
        plt.plot(t_list, this_code_prob_list, label=str(N_layers))

    ## code with t_xyi=t_zi=1
    # # this_code_prob_list = [
    # #     probsucc_poly_fromexpress_conc(t, t, t, code_prob_expr_full) for t
    # #     in t_list]
    # # plt.plot(t_list, this_code_prob_list, 'k--', label='Asymptotic')

    plt.legend()
    plt.show()

    #################################################################
    ############################# OTHER TESTS #######################
    #################################################################

    # ## new graph
    # graph1 = gen_ring_graph(6)
    # gstate1 = GraphState(graph1)
    #
    # # get expression for full decoder
    # decoder1 = LT_FullDecoder(gstate1, in_qubit)
    # code_prob_expr_full1 = get_LTdecoder_succpob_treesearch(decoder1)
    #
    # # get expression for ind X measurement
    # decoder1 = LT_IndMeasDecoder(gstate1, 'X', in_qubit)
    # code_prob_expr_x1 = get_LTdecoder_succpob_treesearch(decoder1)
    #
    # # get expression for ind Y measurement
    # decoder1 = LT_IndMeasDecoder(gstate1, 'Y', in_qubit)
    # code_prob_expr_y1 = get_LTdecoder_succpob_treesearch(decoder1)
    #
    # # get expression for ind Z measurement
    # decoder1 = LT_IndMeasDecoder(gstate1, 'Z', in_qubit)
    # code_prob_expr_z1 = get_LTdecoder_succpob_treesearch(decoder1)
    #
    # ############# plots
    #
    # t_list = np.linspace(0, 1, 30)
    #
    # plt.plot(t_list, t_list, 'k:', label='Direct')
    #
    #
    # conc_code_prob_list_0 = [
    #     conc_prob_full(t, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z, 2) for t
    #     in t_list]
    # plt.plot(t_list, conc_code_prob_list_0, label='graph0')
    #
    # conc_code_prob_list_1 = [
    #     conc_prob_full(t, code_prob_expr_full1, code_prob_expr_x1, code_prob_expr_y1, code_prob_expr_z1, 2) for t
    #     in t_list]
    # plt.plot(t_list, conc_code_prob_list_1, label='graph1')
    #
    #
    # casc_code_prob_list_00 = [
    #     conc_prob_full(t, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z, 3) for t
    #     in t_list]
    # plt.plot(t_list, casc_code_prob_list_00, label='graph0 - graph0')
    #
    # casc_code_prob_list_11 = [
    #     conc_prob_full(t, code_prob_expr_full1, code_prob_expr_x1, code_prob_expr_y1, code_prob_expr_z1, 3) for t
    #     in t_list]
    # plt.plot(t_list, casc_code_prob_list_11, label='graph1 - graph1')
    # #
    # casc_code_prob_list_01 = [
    #     conc_prob_full(t,
    #                    [code_prob_expr_full, code_prob_expr_full1],
    #                    [code_prob_expr_x, code_prob_expr_x1],
    #                    [code_prob_expr_y, code_prob_expr_y1],
    #                    [code_prob_expr_z, code_prob_expr_z1])
    #     for t in t_list]
    # plt.plot(t_list, casc_code_prob_list_01, label='graph0 - graph1')
    #
    # casc_code_prob_list_10 = [
    #     conc_prob_full(t,
    #                    [code_prob_expr_full1, code_prob_expr_full],
    #                    [code_prob_expr_x1, code_prob_expr_x],
    #                    [code_prob_expr_y1, code_prob_expr_y],
    #                    [code_prob_expr_z1, code_prob_expr_z])
    #     for t in t_list]
    # plt.plot(t_list, casc_code_prob_list_10, label='graph1 - graph0')
    #
    # plt.legend()
    # plt.show()
