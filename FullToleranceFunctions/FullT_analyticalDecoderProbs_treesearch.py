from copy import deepcopy, copy
from collections import Counter
from ErrorCorrectionFunctions.EC_DecoderClasses import calculate_syndromes_dictionary_single_ind_meas, \
    calculate_syndromes_dictionary_teleport


################################################################################################
###   FUNCTIONS TO CALCULATE SUCC PROBS OF A DECODER FOR BOTH LOSS AND ERROR CORRECTION      ###
################################################################################################


def get_FullTdecoder_succpob_treesearch(decoder, poss_meas_outcomes=None,
                                        include_direct_meas=False, max_error_num=None):
    if poss_meas_outcomes is None:
        poss_meas_outcomes = {'X': ['X', 'Zind', 'na'], 'Y': ['Y', 'Zind', 'na'],
                              'Z': ['Z', 'na'],
                              'Out': ['Out', 'Zind', 'na']}
    list_running_decs = [decoder]
    list_finished_decs = []

    while list_running_decs:
        # print('\nDoing new steps, current running decoders:', len(list_running_decs), 'and finished', len(list_finished_decs))
        temp_run_decs = []
        temp_finish_decs = []
        for this_dec in list_running_decs:
            # print()
            # print('Looking at decoder with meas_config', this_dec.meas_config)
            # print('Needs new meas_config?', this_dec.new_strategy)
            # print('its strat list is', this_dec.poss_strat_list)
            # print('its meas status is', this_dec.mOUT_OUT, this_dec.mOUT_Z, this_dec.mOUT_na,
            #                  this_dec.mX_X, this_dec.mX_Z, this_dec.mX_na,
            #                  this_dec.mY_Y, this_dec.mY_Z, this_dec.mY_na,
            #                  this_dec.mZ_Z, this_dec.mZ_na)
            # print('its finished status is', this_dec.finished)

            # if there are no possible measurement to do, we have failed and we stop
            if not this_dec.poss_strat_list:
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
                        temp_dec.update_decoder_after_measure(this_out)

                        if temp_dec.finished:
                            temp_finish_decs.append(temp_dec)
                        else:
                            temp_run_decs.append(temp_dec)

        list_running_decs = temp_run_decs
        list_finished_decs = list_finished_decs + temp_finish_decs

    ### Perform error-correction analysis on the decoders that succesffuly finished
    successful_decoders = [this_dec for this_dec in list_finished_decs if this_dec.on_track]
    succ_decs_syndromes_dicts = [
        calculate_syndromes_dictionary_single_ind_meas(this_dec.meas_config[0], this_dec.meas_config[1][0],
                                                       this_dec.meas_config[1][1], this_dec.in_qubit,
                                                       include_direct_meas=include_direct_meas,
                                                       max_error_num=max_error_num) for this_dec in successful_decoders]

    succ_prob_poly_terms = [((len(this_dec.mOUT_OUT), len(this_dec.mOUT_Z), len(this_dec.mOUT_na),
                              len(this_dec.mX_X), len(this_dec.mX_Z), len(this_dec.mX_na),
                              len(this_dec.mY_Y), len(this_dec.mY_Z), len(this_dec.mY_na),
                              len(this_dec.mZ_Z), len(this_dec.mZ_na)
                              ), syndromes_dict_to_tuples(succ_decs_syndromes_dicts[dec_ix]))
                            for dec_ix, this_dec in enumerate(successful_decoders)]

    ## This output is structured as:
    ## {(measurement_types_tuple, decoder_syndromes_tuples): coefficient}
    ## where measurement_types_tuple is the output as in the LT case, and decoder_syndromes_tuples is the syndromes
    ## dictionary for the EC case, converted into tuples (the function tuples_to_syndromes_dict below converts it back
    ## into the dictionary).
    return dict(Counter(succ_prob_poly_terms))


######################################
###   PROBABILITIES CALCULATIONS   ###
######################################

#### Loss-tolerance part
def success_prob_from_poly_expr(t, poly_term, t_xi=1, t_yi=1, t_zi=0):
    t_xyi = max(t_xi, t_yi)
    return ((t * t_xyi) ** poly_term[0]) * \
           (((1 - t) * t_zi) ** poly_term[1]) * \
           (((1 - t) * (1 - t_zi) + t * (1 - t_xyi)) ** poly_term[2]) * \
           ((t * t_xi) ** poly_term[3]) * \
           (((1 - t) * t_zi) ** poly_term[4]) * \
           (((1 - t) * (1 - t_zi) + t * (1 - t_xi)) ** poly_term[5]) * \
           ((t * t_yi) ** poly_term[6]) * \
           (((1 - t) * t_zi) ** poly_term[7]) * \
           (((1 - t) * (1 - t_zi) + t * (1 - t_yi)) ** poly_term[8]) * \
           ((t * (1 - t_zi) + t_zi * (1 - t) + t_zi * t) ** poly_term[9]) * \
           ((1 - t - (1 - t) * t_zi) ** poly_term[10])


#### Error-correction part

# function to calculate the error probability associated to a dictionary with the coefficients associate to different
# probability structures
def calculate_prob_from_struct_coeff_dict(err_prob_structs_coeffs_dict, err_prob_X, err_prob_Y=None,
                                          err_prob_Z=None):
    if err_prob_Y is None:
        err_prob_Y = err_prob_X
    if err_prob_Z is None:
        err_prob_Z = err_prob_X
    temp_prob = 0
    for err_prob_struct in err_prob_structs_coeffs_dict:
        struct_coeff = err_prob_structs_coeffs_dict[err_prob_struct]
        if struct_coeff > 0:
            temp_prob += struct_coeff * ((1 - err_prob_X) ** err_prob_struct[0]) * (err_prob_X ** err_prob_struct[1]) * \
                         ((1 - err_prob_Y) ** err_prob_struct[2]) * (err_prob_Y ** err_prob_struct[3]) * \
                         ((1 - err_prob_Z) ** err_prob_struct[4]) * (err_prob_Z ** err_prob_struct[5])
    return temp_prob


# function that gets the dictionary with the coefficients associated to different probability structures
# when summing multiple of such probabilities
def sum_prob_struct_coeffs_dicts(prob_dict_list):
    temp_struct_coeffs_dict = copy(prob_dict_list[0])
    for temp_prob_dict in prob_dict_list[1:]:
        for prob_struct in temp_prob_dict:
            if prob_struct in temp_struct_coeffs_dict:
                temp_struct_coeffs_dict[prob_struct] += temp_prob_dict[prob_struct]
            else:
                temp_struct_coeffs_dict[prob_struct] = temp_prob_dict[prob_struct]
    return temp_struct_coeffs_dict


### Calculates the error probability given a syndromes lookup dictionary
def error_prob_from_lookup_dict(lookup_dict, err_prob_X, err_prob_Y=None, err_prob_Z=None, num_prob_thresh=1e-12):
    if err_prob_Y is None:
        err_prob_Y = err_prob_X
    if err_prob_Z is None:
        err_prob_Z = err_prob_X
    syndr_probs = dict()
    for syndr in lookup_dict:
        # print('syndr&lookup', syndr, lookup_dict[syndr])
        syndr_probs[syndr] = calculate_prob_from_struct_coeff_dict(
            sum_prob_struct_coeffs_dicts(list(lookup_dict[syndr].values())),
            err_prob_X, err_prob_Y, err_prob_Z)
        # print('prob', syndr_probs[syndr])
    ### Normalize meas. success prob for given syndrome, and perform error correction
    error_probs_syndromes = dict()
    for syndr in lookup_dict:
        if syndr_probs[syndr] > num_prob_thresh:
            # TODO: for teleportation this bit might be modified a bit: different errors provide different
            #  corrections to the Bell state, and they may come with different noises.
            # for error correction, majority voting is assumed here
            error_probs_syndromes[syndr] = min(
                [calculate_prob_from_struct_coeff_dict(lookup_dict[syndr][log_ops_errors],
                                                       err_prob_X, err_prob_Y,
                                                       err_prob_Z) / syndr_probs[syndr]
                 for log_ops_errors in lookup_dict[syndr]])
        else:
            error_probs_syndromes[syndr] = 0
    ### Get final probability
    # return [error_probs_syndromes[syndr] * syndr_probs[syndr] for syndr in lookup_dict]
    return [(error_probs_syndromes[syndr], syndr_probs[syndr]) for syndr in lookup_dict]


#### Combine both Loss-Tolerance and Error-Correction

#### function that takes the, for given noise values, takes the analytical output of the decoder and converts it into a
# list of the type
# [(success prob. for a measurement patter M1,
# [(error prob given syndr1 and M1, proability to measure syndr1 given M1), ... for all syndromes given M1])
# , ... for all measurements]
def prob_structure_from_decoder_analytical_output(analyt_dec_output, t, err_prob_X, err_prob_Y=None, err_prob_Z=None,
                                                  t_xi=1, t_yi=1, t_zi=0, num_prob_thresh=1e-12):
    return [(analyt_dec_output[term]*success_prob_from_poly_expr(t, term[0], t_xi=t_xi, t_yi=t_yi, t_zi=t_zi),
             error_prob_from_lookup_dict(tuples_to_syndromes_dict(term[1]), err_prob_X, err_prob_Y=err_prob_Y,
                                         err_prob_Z=err_prob_Z, num_prob_thresh=num_prob_thresh)
             ) for term in analyt_dec_output]


def transm_and_err_prob_from_prob_struct(prob_struct):
    transm_prob = sum((x[0]*sum((y[1] for y in x[1])) for x in prob_struct))
    error_prob = sum((x[0]*sum((y[0] * y[1] for y in x[1])) for x in prob_struct)) / transm_prob
    return transm_prob, error_prob



###################################
###   OTHER USEFUL FUNCTIONS    ###
###################################


### functions to convert the syndromes dictionaries into tuples to make them hashable
def dict_to_tuples(this_dict):
    return zip(*this_dict.items())


def tuples_to_dict(this_tuple):
    return dict(zip(this_tuple[0], this_tuple[1]))


### Converts syndromes_dicts into hashable tuples
def syndromes_dict_to_tuples(syndr_dict):
    all_syndrs, error_dicts = dict_to_tuples(syndr_dict)
    return (all_syndrs, tuple((x[0], tuple(tuple(dict_to_tuples(y)) for y in x[1])) for x in
                              (tuple(dict_to_tuples(this_error_dict)) for this_error_dict in error_dicts)))


### Converts back tuples into the associated syndromes_dicts
def tuples_to_syndromes_dict(tuples):
    return tuples_to_dict((tuples[0], tuple(
        tuple(tuples_to_dict(y) for y in ((x[0], tuple(tuples_to_dict(y) for y in x[1])) for x in tuples[1])))))


########################################################################################################################
##############################
###          MAIN          ###
##############################

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx
    from FullToleranceFunctions.FullT_Decoders_Classes import FullT_IndMeasDecoder

    from itertools import chain

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    # three graph
    # branching = [2, 2]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    # ## fully connected graph
    # graph = gen_fullyconnected_graph(4)
    # gstate = GraphState(graph)

    ### ring graph
    graph = gen_ring_graph(5)
    gstate = GraphState(graph)

    ##############################################################
    ################## TEST FULL DECODER #########################
    ##############################################################
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    # decod0 = LT_FullDecoder(gstate, in_qubit)
    decod0 = FullT_IndMeasDecoder(gstate, 'X', in_qubit)

    succ_prob_poly_terms = get_FullTdecoder_succpob_treesearch(decod0)

    # print()
    print(succ_prob_poly_terms)
    # print(list(succ_prob_poly_terms.values()))

    def tidyup_succ_prob_poly_terms(succ_prob_poly_terms):
        return dict(zip([x[0] for x in succ_prob_poly_terms], succ_prob_poly_terms.values()))
    print(tidyup_succ_prob_poly_terms(succ_prob_poly_terms))


    transm = 0.9
    error_prob = 0.1

    prob_struct = prob_structure_from_decoder_analytical_output(succ_prob_poly_terms, transm, error_prob)
    print(prob_struct)
    print(transm_and_err_prob_from_prob_struct(prob_struct))




    # ##### Plots
    # gstate.image(input_qubits=[in_qubit])
    # plt.show()
    #
    # t_list = np.linspace(0, 1, 100)
    # used_t_xi = 1.
    # used_t_yi = 1.
    # used_t_zi = 0.
    # succ_probs = [probsucc_poly_fromexpress(t, succ_prob_poly_terms, used_t_xi, used_t_yi, used_t_zi) for t in t_list]
    # plt.plot(t_list, succ_probs, c='blue', label='Analyt.')
    #
    # plt.plot(t_list, t_list, 'k:', label='Direct')
    # plt.legend()
    # plt.show()
