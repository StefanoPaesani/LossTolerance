from copy import deepcopy
from collections import Counter


##############################################################
###   FUNCTIONS TO CALCULATE SUCC PROBS OF A DECODER       ###
##############################################################


def get_LTdecoder_succpob_treesearch(LTdecoder, poss_meas_outcomes=None):
    if poss_meas_outcomes is None:
        poss_meas_outcomes = {'X': ['X', 'Z', 'na'], 'Y': ['X', 'Z', 'na'], 'Z': ['Z', 'na'], 'XYout': ['XYout', 'Z', 'na'],}
    list_running_decs = [LTdecoder]
    list_finished_decs = []

    while list_running_decs:
        # print('\nDoing new steps, current running decoders:', len(list_running_decs), 'and finished', len(list_finished_decs))
        temp_run_decs = []
        temp_finish_decs = []
        for this_dec in list_running_decs:
            # print()
            # print('Looking at decoder with meas_config', this_dec.meas_config)
            # print('its strat list is', this_dec.poss_strat_list)
            # print('its meas status is', this_dec.mOUT_OUT_qbts, this_dec.mOUT_Z_qbts, this_dec.mOUT_na_qbts,
            #       this_dec.mX_X_qbts, this_dec.mX_Z_qbts, this_dec.mX_na_qbts,
            #       this_dec.mY_Y_qbts, this_dec.mY_Z_qbts, this_dec.mY_na_qbts,
            #       this_dec.mZ_Z_qbts, this_dec.mZ_na_qbts,)
            # print('its finished status is', this_dec.finished)

            # if there are no possible measurement to do, we have failed and we stop
            if len(this_dec.poss_strat_list) == 0:
                this_dec.on_track = False
                this_dec.finished = True
                temp_finish_decs.append(this_dec)
            else:
                this_dec.decide_next_meas()
                if this_dec.finished:
                    temp_finish_decs.append(this_dec)
                else:
                    meas_pauli = this_dec.meas_type
                    avail_outcomes = poss_meas_outcomes[meas_pauli]
                    for this_out in avail_outcomes:
                        temp_dec = deepcopy(this_dec)
                        if meas_pauli == 'Z':
                            temp_dec.update_decoder_Zmeas(this_out)
                        elif meas_pauli in ['X', 'Y', 'XYout']:
                            temp_dec.update_decoder_XYmeas(this_out)

                        if temp_dec.finished:
                            temp_finish_decs.append(temp_dec)
                        else:
                            temp_run_decs.append(temp_dec)

        list_running_decs = temp_run_decs
        list_finished_decs = list_finished_decs + temp_finish_decs

    successful_decoders = [this_dec for this_dec in list_finished_decs if this_dec.on_track]
    succ_prob_poly_terms = [(len(this_dec.mOUT_OUT_qbts), len(this_dec.mOUT_Z_qbts), len(this_dec.mOUT_na_qbts),
                             len(this_dec.mX_X_qbts), len(this_dec.mX_Z_qbts), len(this_dec.mX_na_qbts),
                             len(this_dec.mY_Y_qbts), len(this_dec.mY_Z_qbts), len(this_dec.mY_na_qbts),
                             len(this_dec.mZ_Z_qbts), len(this_dec.mZ_na_qbts)) for this_dec in successful_decoders]
    return dict(Counter(succ_prob_poly_terms))


# Function that converts the polynomial expression into the success probability as a function of:
# transmission t, and indirect measurement probabilities p_xyi, p_zi
# The terms in the expression are in the order: (OUT_OUT, OUT_Z, OUT_na, X_X, X_Z, X_na, Y_Y, Y_Z, Y_na, Z_Z, Z_na)
def probsucc_poly_fromexpress(t, poly_express, t_xi=1, t_yi=1, t_zi=0):
    t_xyi = max(t_xi, t_yi)
    return sum(
        [poly_express[term] *
         ((t*t_xyi)**term[0]) *
         (((1-t)*t_zi)**term[1]) *
         (((1-t)*(1-t_zi) + t*(1-t_xyi))**term[2]) *

         ((t * t_xi) ** term[3]) *
         (((1 - t) * t_zi) ** term[4]) *
         (((1 - t) * (1 - t_zi) + t * (1 - t_xi)) ** term[5]) *

         ((t * t_yi) ** term[6]) *
         (((1 - t) * t_zi) ** term[7]) *
         (((1 - t) * (1 - t_zi) + t * (1 - t_yi)) ** term[8]) *

         ((t+(1-t)*t_zi)**term[9]) *
         ((1-t-(1-t)*t_zi)**term[10])
         for term in poly_express])


########################################################################################################################
##############################
###          MAIN          ###
##############################

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx
    from LossToleranceFunctions.LT_Decoders_Classes import LT_FullDecoder, LT_IndMeasDecoder

    from itertools import chain


    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


    branching = None

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    # three graph
    # branching = [2, 2]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    ## fully connected graph
    graph = gen_fullyconnected_graph(7)
    gstate = GraphState(graph)

    # ### ring graph
    # graph = gen_ring_graph(4)
    # gstate = GraphState(graph)

    # # ## Two-level Tree graph [2 ,2]
    # graph_nodes = list(range(6))
    # graph_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    ##############################################################
    ################## TEST FULL DECODER #########################
    ##############################################################

    # decod0 = LT_FullDecoder(gstate, in_qubit)
    decod0 = LT_IndMeasDecoder(gstate, 'X', in_qubit)

    succ_prob_poly_terms = get_LTdecoder_succpob_treesearch(decod0)

    print(succ_prob_poly_terms)

    ##### Plots
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    t_list = np.linspace(0, 1, 100)
    used_t_xi = 1.
    used_t_yi = 1.
    used_t_zi = 0.
    succ_probs = [probsucc_poly_fromexpress(t, succ_prob_poly_terms, used_t_xi, used_t_yi, used_t_zi) for t in t_list]
    plt.plot(t_list, succ_probs, c='blue', label='Analyt.')

    plt.plot(t_list, t_list, 'k:', label='Direct')
    plt.legend()
    plt.show()
