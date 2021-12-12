from CodesFunctions.LTCodeClass import LTCode
from CodesFunctions.graphs import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    ##################################
    #####     INITIALISATION     #####
    ##################################
    ##### Define graph dimensions & calculation parameters
    nrows = 3
    nlayers = 3

    alpha = 3

    num_MC_trials = 1000   # DECREASE THIS IF SIMULATIONS TAKE TOO LONG
    loss_prob_list = np.linspace(0, 1, 10)

    ##### initialise lists

    codes_LT_probs_list = []
    labels = []

    ####################################
    #####     GRAPH DEFINITION     #####
    ####################################

    ########## Lattice graph encodings
    # encode_graph = gen_crazy_graph(nrows, nlayers)
    # encode_graph = gen_square_lattice_graph(nrows, nlayers)
    # encode_graph = gen_triangular_lattice_graph(nrows, nlayers)
    # encode_graph = gen_hexagonal_lattice_graph(nrows, nlayers)
    # encode_graph = gen_multiwire_graph(nrows, nlayers)

    # in_nodes = list(range(nrows))
    # out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))

    ########## Fully-connected graph econding
    nqbts = nrows * nlayers
    encode_graph = gen_fullyconnected_graph(nqbts)
    in_nodes = list(range(int(nqbts / 2)))
    out_nodes = list(range(int(nqbts / 2), nqbts))


    ##################
    ### START TEST ###
    ##################

    mycode = LTCode(encode_graph, in_nodes, out_nodes)

    print("Calculating all teleportation measurements")
    start_time = time.time()
    tele_meas_SPalg = mycode.SPalgorithm_valid_teleportation_meas(max_m_increase=alpha, test_inouts=True,
                                                                  exclude_input_ys=True, return_evolution=False)
    end_time = time.time()
    print("Time passed for SP algorithm with alpha", alpha, ":", end_time - start_time, "s, found", len(tele_meas_SPalg)
          , "measurements")

    #############################################
    ### Monte-Carlo Loss-Tolerance Estimation ###
    #############################################

    ####################### Type 1

    weight_fact = 0
    max_tree_depth = 0
    follow_curr_best = False

    this_label = 'w:'+str(weight_fact)+'; dpt:'+str(max_tree_depth)+'; fcb:'+str(follow_curr_best)

    print('\nStarting unheralded loss tolerance probability scan - ' + this_label)
    start_time = time.time()

    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=follow_curr_best,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    codes_LT_probs_list.append(tele_succ_prob_list)
    labels.append(this_label)


    ####################### Type 2

    weight_fact = 4
    max_tree_depth = 0
    follow_curr_best = False

    this_label = 'w:'+str(weight_fact)+'; dpt:'+str(max_tree_depth)+'; fcb:'+str(follow_curr_best)

    print('\nStarting unheralded loss tolerance probability scan - ' + this_label)
    start_time = time.time()

    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=follow_curr_best,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    codes_LT_probs_list.append(tele_succ_prob_list)
    labels.append(this_label)


    ####################### Type 3

    weight_fact = 4
    max_tree_depth = 3
    follow_curr_best = False

    this_label = 'w:' + str(weight_fact) + '; dpt:' + str(max_tree_depth) + '; fcb:' + str(follow_curr_best)

    print('\nStarting unheralded loss tolerance probability scan - ' + this_label)
    start_time = time.time()

    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=follow_curr_best,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    codes_LT_probs_list.append(tele_succ_prob_list)
    labels.append(this_label)



    ####################### Type 4

    weight_fact = 4
    max_tree_depth = 3
    follow_curr_best = True

    this_label = 'w:' + str(weight_fact) + '; dpt:' + str(max_tree_depth) + '; fcb:' + str(follow_curr_best)

    print('\nStarting unheralded loss tolerance probability scan - ' + this_label)
    start_time = time.time()

    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=follow_curr_best,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    codes_LT_probs_list.append(tele_succ_prob_list)
    labels.append(this_label)


    ####################### Type 5

    weight_fact = 4
    max_tree_depth = 5
    follow_curr_best = False

    this_label = 'w:' + str(weight_fact) + '; dpt:' + str(max_tree_depth) + '; fcb:' + str(follow_curr_best)

    print('\nStarting unheralded loss tolerance probability scan - ' + this_label)
    start_time = time.time()

    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=follow_curr_best,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    codes_LT_probs_list.append(tele_succ_prob_list)
    labels.append(this_label)


    ####################### Type 6

    weight_fact = 4
    max_tree_depth = 5
    follow_curr_best = True

    this_label = 'w:' + str(weight_fact) + '; dpt:' + str(max_tree_depth) + '; fcb:' + str(follow_curr_best)

    print('\nStarting unheralded loss tolerance probability scan - ' + this_label)
    start_time = time.time()

    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=follow_curr_best,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    codes_LT_probs_list.append(tele_succ_prob_list)
    labels.append(this_label)




    ####################### Type 7

    weight_fact = 4
    max_tree_depth = mycode.res_graph_num_nodes
    follow_curr_best = True

    this_label = 'w:' + str(weight_fact) + '; dpt:' + str(max_tree_depth) + '; fcb:' + str(follow_curr_best)

    print('\nStarting unheralded loss tolerance probability scan - ' + this_label)
    start_time = time.time()

    tele_succ_prob_list = mycode.Unheralded_loss_teleport_prob_MC_estimation(tele_meas_SPalg, loss_prob_list,
                                                                             follow_curr_best=follow_curr_best,
                                                                             weight_fact=weight_fact,
                                                                             max_tree_depth=max_tree_depth,
                                                                             MC_trials=num_MC_trials)
    end_time = time.time()
    print("Completed in:", end_time - start_time)

    codes_LT_probs_list.append(tele_succ_prob_list)
    labels.append(this_label)

    ####################################
    ########     PLOT DATA     #########
    ####################################


    plt.figure()
    for stategy_ix, tele_succ_prob_list in enumerate(codes_LT_probs_list):
        plt.plot(loss_prob_list, tele_succ_prob_list, label=labels[stategy_ix])
    plt.xlabel('Loss-per-photon probability')
    plt.ylabel('Teleportation Probability')
    plt.title('One-qubit Teleportation - Unheralded Loss')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend(loc='upper right')
    plt.show()



    ##############################################################

    plt.figure()
    mycode.image(with_labels=True)
    plt.show()