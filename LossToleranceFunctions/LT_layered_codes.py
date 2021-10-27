from LossToleranceFunctions.LT_Decoders_Classes import LT_FullDecoder, LT_IndMeasDecoder
from LossToleranceFunctions.LT_analyticalDecodersProbs_treesearch import get_LTdecoder_succpob_treesearch, \
    probsucc_poly_fromexpress


###############################################
####### FUNCTIONS FOR CASCADED CODES ##########
###############################################

def probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, poly_express):
    return probsucc_poly_fromexpress(t, t_xyi, t_xyi, t_zi, poly_express)


def cascade_prob_X(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return 1
    else:
        p_x = cascade_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_x)


def cascade_prob_Y(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return 1
    else:
        p_x = cascade_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_y)


def cascade_prob_Z(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return 0
    else:
        p_x = cascade_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_z)


def cascade_prob_full(t, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if N_layers == 0:
        return t
    else:
        p_x = cascade_prob_X(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        p_y = cascade_prob_Y(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_xyi = max(p_x, p_y)
        t_zi = cascade_prob_Z(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_casc(t, t_xyi, t_zi, code_prob_expr_full)


###################################################
####### FUNCTIONS FOR CONCATENATED CODES ##########
###################################################

def probsucc_poly_fromexpress_conc(t_xi, t_yi, t_zi, poly_express):
    t_xyi = max(t_xi, t_yi)
    return sum(
        [poly_express[term] *
         (t_xyi**term[0]) *
         ((1-t_xyi)**term[2]) *

         (t_xi ** term[3]) *
         ((1 - t_xi) ** term[5]) *

         (t_yi ** term[6]) *
         ((1 - t_yi) ** term[8]) *

         (t_zi**term[9]) *
         ((1-t_zi)**term[10])
         for term in poly_express])


def conc_prob_X(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return t
    else:
        t_xi = conc_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_yi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_zi = conc_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_conc(t_xi, t_yi, t_zi, code_prob_expr_x)


def conc_prob_Y(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return t
    else:
        t_xi = conc_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_yi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_zi = conc_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_conc(t_xi, t_yi, t_zi, code_prob_expr_y)


def conc_prob_Z(t, layer_ix, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if layer_ix == N_layers:
        return t
    else:
        t_xi = conc_prob_X(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_yi = conc_prob_Y(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_zi = conc_prob_Z(t, layer_ix + 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_conc(t_xi, t_yi, t_zi, code_prob_expr_z)


def conc_prob_full(t, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z):
    if N_layers == 0:
        return t
    else:
        t_xi = conc_prob_X(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_yi = conc_prob_Y(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        t_zi = conc_prob_Y(t, 1, N_layers, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
        return probsucc_poly_fromexpress_conc(t_xi, t_yi, t_zi, code_prob_expr_full)


########################################################################################################################
##############################
###          MAIN          ###
##############################

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx

    # from itertools import chain

    branching = None

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    ## three graph
    # branching = [3]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    ### fully connected graph
    # graph = gen_fullyconnected_graph(7)
    # gstate = GraphState(graph)

    ### ring graph
    graph = gen_ring_graph(5)
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
    # num_layers_list = [0, 1, 2, 3, 4, 5]
    # # num_layers_list = [0, 1, 2, 15]
    #
    # plt.plot(t_list, t_list, 'k:', label='Direct')
    # for N_layers in num_layers_list:
    #     this_code_prob_list = [
    #         cascade_prob_full(t, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
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




    # ##### Plots
    # gstate.image(input_qubits=[in_qubit])
    # plt.show()
    #
    # t_list = np.linspace(0, 1, 30)
    #
    # num_layers_list = [0, 1]
    # # num_layers_list = [0, 1, 2, 15]
    #
    # plt.plot(t_list, t_list, 'k:', label='Direct')
    # for N_layers in num_layers_list:
    #     this_code_prob_list = [
    #         conc_prob_full(t, N_layers, code_prob_expr_full, code_prob_expr_x, code_prob_expr_y, code_prob_expr_z)
    #         for t in t_list]
    #     plt.plot(t_list, this_code_prob_list, label=str(N_layers))

    # ## code with t_xyi=t_zi=1
    # this_code_prob_list = [
    #     probsucc_poly_fromexpress_conc(t, t, t, code_prob_expr_full) for t
    #     in t_list]
    # plt.plot(t_list, this_code_prob_list, 'k--', label='Asymptotic')
    #
    # plt.legend()
    # plt.show()


    #################################################################
    ############################# OTHER TESTS #######################
    #################################################################

    t_list = np.linspace(0, 1, 30)

    plt.plot(t_list, t_list, 'k:', label='Direct')

    casc_code_prob_list = [
        probsucc_poly_fromexpress_casc(t, 1, 0, code_prob_expr_full) for t
        in t_list]
    plt.plot(t_list, casc_code_prob_list, 'r', label='Cascaded')

    conc_code_prob_list = [
        probsucc_poly_fromexpress_conc(t, t, t, code_prob_expr_full) for t
        in t_list]
    plt.plot(t_list, conc_code_prob_list, 'b', label='Concatenated')

    plt.legend()
    plt.show()