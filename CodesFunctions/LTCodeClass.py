from CodesFunctions.local_transformations import pauli_measurement_on_code

import qecc as q

from copy import deepcopy
from itertools import chain, combinations, combinations_with_replacement, product
from functools import reduce
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

PauliOps = ['I', 'X', 'Y', 'Z']


##############################
###### CLASS DEFINITION ######
##############################

class LTCode(object):
    r"""
    Class representing a code for loss-tolerant encodings in graph states.

    :param encoding_graph: The graph representing the resource graph state to encode the logical state.
    :type encoding_graph: :class:`nx.Graph`
    :param input_vertices: Iterator over the vertices of the encoding graph to be used for the input encoding
    :param output_vertices: Iterator over the vertices of the encoding graph to be used for the output encoding
    """

    def __init__(self, encoding_graph, input_vertices, output_vertices):
        if len(input_vertices) == 0:
            raise ValueError("At least one qubit and one input mode associated to it are needed")

        if isinstance(input_vertices[0], int):
            in_vtx = [input_vertices]
        else:
            in_vtx = input_vertices

        if len(output_vertices) > 0:
            if isinstance(output_vertices[0], int):
                out_vtx = [output_vertices]
            else:
                out_vtx = output_vertices
        else:
            out_vtx = output_vertices

        if (len(out_vtx) > 0) and (len(in_vtx) != len(out_vtx)):
            raise ValueError("Number of input qubits needs to be the same as number of output qubits")
        self.num_logical_qbits = len(in_vtx)

        self.res_graph = encoding_graph

        self.res_inputs = in_vtx
        self.res_outputs = out_vtx

        self.res_inputs_flat = [item for sublist in in_vtx for item in sublist]
        self.res_outputs_flat = [item for sublist in out_vtx for item in sublist]

        # check that there is no overlap between the output nodes:
        for ix_1 in range(len(self.res_outputs) - 1):
            for ix_2 in range(ix_1 + 1, len(self.res_outputs)):
                # print(self.res_outputs[ix_1], self.res_outputs[ix_2])
                if any([i in self.res_outputs[ix_1] for i in self.res_outputs[ix_2]]):
                    raise ValueError("The output node sets", ix_1, "and", ix_2, "have non-empty intersection.")

        # check that there is no overlap between the input nodes:
        for ix_1 in range(len(self.res_inputs) - 1):
            for ix_2 in range(ix_1 + 1, len(self.res_inputs)):
                if any(i in self.res_inputs[ix_1] for i in self.res_inputs[ix_2]):
                    raise ValueError("The input node sets", ix_1, "and", ix_2, "have non-empty intersection.")

        # plt.subplot(111)
        # GraphState(encoding_graph).image(with_labels=True)
        # plt.show()

        self.res_graph_nodes = encoding_graph.nodes()
        self.res_graph_edges = encoding_graph.edges()
        self.res_graph_num_nodes = len(self.res_graph_nodes)
        self.nqubits = self.res_graph_num_nodes + len(in_vtx) + len(out_vtx)
        # TODO: maybe this check can be avoided, but for now let's keep this contraint on the graph definition
        if list(self.res_graph_nodes) != list(range(self.res_graph_num_nodes)):
            raise ValueError("The nodes of the graph must be labelled as [0,1,2,...,n-1]")

        # copy graph
        total_graph = deepcopy(encoding_graph)

        # check if graph layers are defined in the resource graph
        check_layers_def = all(["layer" in this_node[1] for this_node in self.res_graph.nodes(data=True)])
        if check_layers_def:
            nodes_layers = [this_node[1]["layer"] for this_node in self.res_graph.nodes(data=True)]
            in_nodes_layer = min(nodes_layers) - 1
            out_nodes_layer = max(nodes_layers) + 1

        ### add input nodes and edges
        self.input_nodes = list(range(self.res_graph_num_nodes, self.res_graph_num_nodes + self.num_logical_qbits))
        in_edges = []
        for in_node_ix, this_node in enumerate(self.input_nodes):
            if check_layers_def:
                total_graph.add_node(this_node, layer=in_nodes_layer)
            else:
                total_graph.add_node(this_node)
            these_edges = list(product([this_node], in_vtx[in_node_ix]))
            total_graph.add_edges_from(these_edges)
            in_edges.append(list(these_edges))
        self.input_edges = in_edges

        ### add output nodes and edges
        self.output_nodes = list(
            range(self.res_graph_num_nodes + len(self.res_outputs), self.res_graph_num_nodes +
                  2 * len(self.res_outputs))
        )
        out_edges = []
        if len(output_vertices) > 0:
            for out_node_ix, this_node in enumerate(self.output_nodes):
                if check_layers_def:
                    total_graph.add_node(this_node, layer=out_nodes_layer)
                else:
                    total_graph.add_node(this_node)
                these_edges = list(product([this_node], out_vtx[out_node_ix]))
                total_graph.add_edges_from(these_edges)
                out_edges.append(these_edges)
        self.output_edges = out_edges

        # define total graph of the loss-tolerant graph code
        self.code_graph = total_graph

        # get the logical operators of the graph code
        self.logical_ops = self.get_logical_ops()

        # get the stabilizer generators for the graph code
        self.stab_gens = self.get_code_stabilizer_gens()

    ################
    ### PRINTING ###
    ################

    def image(self, with_labels=True, position_nodes=None, xdist=0.5, rescale_fact=0.9):
        """
        Produces a matplotlib image of the graph associated to the loss-tolerant code.
        """
        pos_nodes = position_nodes
        if pos_nodes is None:
            # checks if all nodes have attribute "layer", and, if they do, plot the graph using multilayer_layout
            if all(["layer" in this_node[1] for this_node in self.code_graph.nodes(data=True)]):
                pos_nodes = nx.multipartite_layout(self.code_graph, subset_key="layer")
            else:
                # pos_nodes_res = nx.spring_layout(self.res_graph)
                pos_nodes_res = nx.kamada_kawai_layout(self.res_graph)
                #### add a rotation of pi/2 to kamada_kawai_layout
                pos_nodes_res = dict(zip(pos_nodes_res.keys(),
                                         map(lambda x: rotate(x * rescale_fact, np.pi / 2.), pos_nodes_res.values())))
                pos_res_nodes_xylist = np.array(list(pos_nodes_res.values()))
                pos_res_nodes_xs = pos_res_nodes_xylist[:, 0]
                pos_res_nodes_ys = pos_res_nodes_xylist[:, 1]
                max_x = np.max(pos_res_nodes_xs)
                min_x = np.min(pos_res_nodes_xs)
                max_y = np.max(pos_res_nodes_ys)
                min_y = np.min(pos_res_nodes_ys)

                pos_nodes = pos_nodes_res
                ydist = (max_y - min_y) / self.num_logical_qbits
                for inoutrow_ix in range(self.num_logical_qbits):
                    pos_nodes[self.input_nodes[inoutrow_ix]] = (min_x - xdist, min_y + ydist * (inoutrow_ix + 0.5))
                    pos_nodes[self.output_nodes[inoutrow_ix]] = (max_x + xdist, min_y + ydist * (inoutrow_ix + 0.5))

        # draw resource graph nodes & edges
        options_nodes = {"node_size": 100, "alpha": 1, "node_color": "navy"}
        nx.draw_networkx_nodes(self.code_graph, pos_nodes, nodelist=self.res_graph_nodes, **options_nodes)
        options_edges = {"width": 1.0, "alpha": 0.8, "edge_color": "black"}
        nx.draw_networkx_edges(self.code_graph, pos_nodes, edgelist=self.res_graph_edges, **options_edges)

        # draw input nodes & edges
        for in_idx in range(self.num_logical_qbits):
            options_nodes = {"node_size": 150, "alpha": 1, "node_color": "darkred"}
            nx.draw_networkx_nodes(self.code_graph, pos_nodes, nodelist=[self.input_nodes[in_idx]], **options_nodes)
            options_edges = {"width": 1.5, "alpha": 0.8, "edge_color": "darkred"}
            nx.draw_networkx_edges(self.code_graph, pos_nodes, edgelist=self.input_edges[in_idx], **options_edges)

        # draw output nodes & edges
        if len(self.res_outputs) > 0:
            for out_idx in range(self.num_logical_qbits):
                options_nodes = {"node_size": 150, "alpha": 1, "node_color": "mediumblue"}
                nx.draw_networkx_nodes(self.code_graph, pos_nodes, nodelist=[self.output_nodes[out_idx]],
                                       **options_nodes)
                options_edges = {"width": 1.5, "alpha": 0.8, "edge_color": "navy"}
                nx.draw_networkx_edges(self.code_graph, pos_nodes, edgelist=self.output_edges[out_idx], **options_edges)

        # draw node labels
        if with_labels:
            nx.draw_networkx_labels(self.code_graph, pos_nodes, font_color='white', font_size=8)
        plt.axis("off")

    ############################################
    ### CODE STABILIZERS & LOGICAL OPERATORS ###
    ############################################

    def get_logical_ops(self):
        """
        Generates the 2k logical operators for the code
        """
        log_Z_list = []
        log_X_list = []
        for in_node in self.input_nodes:
            # define logical X operators
            log_X_list.append(
                q.Pauli.from_sparse({in_node: 'Z'}, nq=self.nqubits)
            )

            # define logical Z operators
            neighbour_nodes = list(self.code_graph.neighbors(in_node))
            pauli_z_dict = dict(zip(neighbour_nodes, ['Z' for i in range(len(neighbour_nodes))]))
            pauli_z_dict[in_node] = 'X'
            log_Z_list.append(
                q.Pauli.from_sparse(pauli_z_dict, nq=self.nqubits)
            )
        # TODO: Here I followed the notation on Sam's paper, but I actually think the notation on Xs and Zs is reversed
        #  in that paper and should therefore be adjusted here
        return [log_Z_list, log_X_list]

    def get_code_stabilizer_gens(self):
        """
        Generates the N-k graph code stabilizer generators
        """

        stab_gens = []
        nqubits = self.nqubits

        sorted_nodes = list(sorted(self.code_graph.nodes()))
        for node_ix in sorted_nodes:
            if node_ix not in self.input_nodes:
                stab_dict = {sorted_nodes.index(node_ix): 'X'}
                for ngb_node_ix in sorted(self.code_graph.neighbors(node_ix)):
                    stab_dict[sorted_nodes.index(ngb_node_ix)] = 'Z'
                this_stab = q.Pauli.from_sparse(stab_dict, nq=nqubits)
                stab_gens.append(this_stab)
        return stab_gens

    def get_code_stabilizer_gens_labelled(self):
        """
        Generates the N-k graph code stabilizer generators, labelled according to the associated node
        """

        stab_gens_labelled = {}
        nqubits = self.nqubits

        sorted_nodes = list(sorted(self.code_graph.nodes()))
        for node_ix in sorted_nodes:
            if node_ix not in self.input_nodes:
                stab_dict = {sorted_nodes.index(node_ix): 'X'}
                for ngb_node_ix in sorted(self.code_graph.neighbors(node_ix)):
                    stab_dict[sorted_nodes.index(ngb_node_ix)] = 'Z'
                this_stab = q.Pauli.from_sparse(stab_dict, nq=nqubits)
                stab_gens_labelled[node_ix] = this_stab
        return stab_gens_labelled

    #########################################
    ### LOCAL OPERATIONS AND MEASUREMENTS ###
    #########################################

    def perform_local_measurement(self, pauli_measurement, outcome=0):
        """
        Calculates the stabilizer generators and logical operators of the code after a local Pauli measurement
        """
        return pauli_measurement_on_code(pauli_measurement, self.stab_gens, self.logical_ops, outcome=outcome)

    def update_code_after_measurement(self, pauli_measurement, outcome=0):
        """
        Update the stabilizer generators and logical operators of the code upon applying a local Pauli measurement
        """
        self.stab_gens, self.logical_ops = self.perform_local_measurement(pauli_measurement, outcome=outcome)

    def check_logical_op_islocal_on_ouputs(self, logical_op, exclude_input_ys=True):
        """
        Check if a logical operator acts on one and only one output qubit, which is a necessary condition for validity
        If exclude_input_ys = True, it also excludes cases where Ys are present in the input qubits, to get rid of a
        2^(num_logical_qubits) degeneracy in the valid logical operators (all giving valid measurements equivalent
        up to local Clifford S operations)
        """
        # counts how many 'I' are on the output qubits
        if (logical_op.op[(-1) * self.num_logical_qbits:]).count('I') == (self.num_logical_qbits - 1):
            if exclude_input_ys:
                if 'Y' in logical_op.op[(-2) * self.num_logical_qbits:(-1) * self.num_logical_qbits]:
                    return False
                else:
                    return True
            else:
                return True
        else:
            return False

    def get_all_valid_logical_ops(self, trivial_stab_test=True, exclude_input_ys=True, test_inouts=False):
        """
        Get all valid logical operators by multiplication with all code stabilizers
        """
        # If trivial_stab_test is True, we use only non-trivial stabilizers,
        # else we use all the possible 2^N stabilizers.
        if trivial_stab_test:
            used_stabilizers = self.all_nontrivial_stabilizers_fullsearch(test_inouts=test_inouts)
        else:
            used_stabilizers = list(q.from_generators(self.stab_gens))
        [Z_log_ops, X_log_ops] = self.logical_ops

        # wrapped in a way that avoids double calculations of this_stab * this_log_op
        all_Z_log_ops = [[y for y in
                          (this_stab * this_log_op for this_stab in used_stabilizers)
                          if self.check_logical_op_islocal_on_ouputs(y, exclude_input_ys=exclude_input_ys)
                          ]
                         for this_log_op in Z_log_ops]

        # wrapped in a way that avoids double calculations of this_stab * this_log_op
        all_X_log_ops = [[y for y in
                          (this_stab * this_log_op for this_stab in used_stabilizers)
                          # X ops cannot have Ys on inputs by construction, so setting exclude_input_ys=False always.
                          if self.check_logical_op_islocal_on_ouputs(y, exclude_input_ys=False)
                          ]
                         for this_log_op in X_log_ops]
        return all_Z_log_ops, all_X_log_ops

    def get_all_Pauli_measurements(self):
        """
        Generate all possible Pauli measurements on the resource graph nodes, X on the inputs, and I on outputs
        """
        temp_Pauli_meas = list(q.pauli_group(self.res_graph_num_nodes))
        for this_Pauli in temp_Pauli_meas:
            for i in range(self.num_logical_qbits):
                this_Pauli.op = this_Pauli.op + 'X'
            for i in range(self.num_logical_qbits):
                this_Pauli.op = this_Pauli.op + 'I'
        return temp_Pauli_meas

    ###############################
    #### TELEPORTATION CHECKS  ####
    ###############################

    def verify_teleportation_condition(self, logical_ops_list):
        z_logops = logical_ops_list[0]
        x_logops = logical_ops_list[1]
        all_logops = z_logops + x_logops
        nonoutput_nodes = list(self.res_graph_nodes) + list(self.input_nodes)
        # check that they commute on all non-output nodes
        pauli_meas_ops = {}
        for node_ix in nonoutput_nodes:
            check_nonId = True
            this_op = 'I'
            # print("Node:", node_ix)
            for log_op in all_logops:
                temp_op = log_op.op[node_ix]
                # print("Node:", node_ix, "; Op:", log_op, "; TempOp:", temp_op,
                #       "; check_nonId:", check_nonId, "; this_op:", this_op)
                if temp_op != 'I':
                    # If this is the first found logical operator with non-identity operator on this qubit,
                    # then set the operator for this qubit, and turn check_nonId to False.
                    if check_nonId:
                        check_nonId = False
                        this_op = temp_op
                    # If a non-identity operator was already found, check if this operator has the same operator as
                    # the previous one. If they don't they anticommute and so we exit.
                    else:
                        if temp_op != this_op:
                            return False,

            pauli_meas_ops[node_ix] = this_op

        # check that they act on single output nodes, and identity on all others
        for this_logop in all_logops:
            if not self.check_logical_op_islocal_on_ouputs(this_logop):
                return False,

        # check anticommutation relations
        for out_qb_ix in range(self.num_logical_qbits):
            this_node_ix = self.output_nodes[out_qb_ix]
            if single_qubit_commute(z_logops[out_qb_ix], x_logops[out_qb_ix], this_node_ix) == 0:
                return False,

        pauli_meas = q.Pauli.from_sparse(pauli_meas_ops, self.nqubits)
        return True, pauli_meas, logical_ops_list

    def all_nontrivial_stabilizers_fullsearch(self, test_inouts=True):
        """
        Find all stabilizers that act non-trivially, and have overlaps on the input and output nodes,
        via exhaustive search.
        """
        # get all combinations of nodes, which possess at lead one node in the encoding and one in the decoding nodes.
        stab_nodes_combs = powerset_nonempty(list(self.res_graph_nodes) + list(self.output_nodes))
        # get the adjacency matrix of graph, to be used to test triviality.
        adj_mat = nx.adjacency_matrix(self.code_graph, nodelist=sorted(self.code_graph.nodes())).todense()
        # Of the combinations on nodes in stab_nodes_combs keep only those that pass the triviality test
        nontrivial_stabnodes = [nodes_set for nodes_set in stab_nodes_combs
                                if self.test_stabilizer_isnontrivial(nodes_set, adj_mat, test_inouts=test_inouts)]
        # For the nontrivial combinations of nodes multiplity the associated stab generators to get
        # the nontrivial stabilizers
        labelles_stab_gens = self.get_code_stabilizer_gens_labelled()
        nontrivial_stabs = [reduce(lambda x, y: x * y, [labelles_stab_gens[ix] for ix in this_nontriv_stabnodes]) for
                            this_nontriv_stabnodes in nontrivial_stabnodes]
        return nontrivial_stabs

    def test_stabilizer_isnontrivial(self, stab_nodes, graph_adj_mat, test_inouts=True):
        stab_nodes_array = np.zeros(self.nqubits, dtype=int)
        for node_ix in stab_nodes:
            stab_nodes_array[node_ix] = 1
        # neighbourhood includes the stab_nodes plus all
        neighbourhood = np.nonzero((graph_adj_mat @ stab_nodes_array) + stab_nodes_array)[1]
        # Check if neighbourhood overlaps with the nodes for the input encoding and output decoding.
        # If it doesn't, the stabilizer cannot provide a valid logical operator, so return False.
        if test_inouts:
            if set(neighbourhood).isdisjoint(self.res_inputs_flat) or \
                    set(neighbourhood).isdisjoint(self.res_outputs_flat):
                return False
        # The subgraph includes all nodes in the neighborhood, but only edges connecting some node in the neighbourhood
        # to one in stab_nodes: these are the only ones that determine whether the support between possible
        # substabilizers overlap or not.
        num_nodes_neigh = len(neighbourhood)
        stab_nodes_pos_in_neigh = np.where(list(map(lambda x: x in stab_nodes, neighbourhood)))[0]
        truncated_subgraph_adj_mat = np.zeros((num_nodes_neigh, num_nodes_neigh))
        truncated_subgraph_adj_mat[stab_nodes_pos_in_neigh] = graph_adj_mat[np.ix_(stab_nodes, neighbourhood)]
        truncated_subgraph_adj_mat[:, stab_nodes_pos_in_neigh] = graph_adj_mat[np.ix_(neighbourhood, stab_nodes)]
        # check if obtained subgraph is connected
        if nx.is_connected(nx.from_numpy_matrix(truncated_subgraph_adj_mat)):
            return True
        else:
            return False

    def full_search_valid_teleportation_meas(self, trivial_stab_test=True, return_logops=False, exclude_input_ys=True):

        all_logops_z, all_logops_x = self.get_all_valid_logical_ops(trivial_stab_test=trivial_stab_test,
                                                                    exclude_input_ys=exclude_input_ys)

        all_z_logops_combs = product(*all_logops_z)
        all_x_logops_combs = product(*all_logops_x)
        all_logops_combs = product(all_z_logops_combs, all_x_logops_combs)

        if return_logops:
            all_valid_tele_meas = [(y[1].op, y[2]) for y in
                                   (self.verify_teleportation_condition(logop_comb, ) for logop_comb in
                                    all_logops_combs)
                                   if y[0]]
        else:
            all_valid_tele_meas = [y[1].op for y in
                                   (self.verify_teleportation_condition(logop_comb, ) for logop_comb in
                                    all_logops_combs)
                                   if y[0]]
        return list(set(all_valid_tele_meas))

    ##################################################
    #### NEW ALGORITHM FOR MORE EFFICIENT SEARCH  ####
    ##################################################

    def find_stabs_given_node_cardinality(self, m, adj_mat, test_inouts=True):
        """
        Find which sets of nodes, with a given cardinality m, provide non-trivial stabilizers
        """
        stab_nodes_combs = combinations(list(self.res_graph_nodes) + list(self.output_nodes), m)
        # Of the combinations on nodes in stab_nodes_combs keep only those that pass the triviality test
        nontrivial_stabnodes = [nodes_set for nodes_set in stab_nodes_combs
                                if self.test_stabilizer_isnontrivial(nodes_set, adj_mat, test_inouts=test_inouts)]
        # For the nontrivial combinations of nodes multiplity the associated stab generators to get
        # the nontrivial stabilizers
        labelles_stab_gens = self.get_code_stabilizer_gens_labelled()
        nontrivial_stabs = [reduce(lambda x, y: x * y, [labelles_stab_gens[ix] for ix in this_nontriv_stabnodes]) for
                            this_nontriv_stabnodes in nontrivial_stabnodes]
        return nontrivial_stabs

    def find_new_logops_for_node_cardinality(self, m, adj_mat, test_inouts=True, exclude_input_ys=True):
        """
        finds the non-trivial logical operators associated to node sets with given cardinality.
        """
        new_stabilizers = self.find_stabs_given_node_cardinality(m, adj_mat, test_inouts=test_inouts)
        [Z_log_ops, X_log_ops] = self.logical_ops
        # wrapped in a way that avoids double calculations of this_stab * this_log_op
        all_Z_log_ops_new = [[y for y in
                              (this_stab * this_log_op for this_stab in new_stabilizers)
                              if self.check_logical_op_islocal_on_ouputs(y, exclude_input_ys=exclude_input_ys)
                              ]
                             for this_log_op in Z_log_ops]
        # wrapped in a way that avoids double calculations of this_stab * this_log_op
        all_X_log_ops_new = [[y for y in
                              (this_stab * this_log_op for this_stab in new_stabilizers)
                              # X ops cannot have Ys on inputs by construction,
                              # so setting exclude_input_ys=False always.
                              if self.check_logical_op_islocal_on_ouputs(y, exclude_input_ys=False)
                              ]
                             for this_log_op in X_log_ops]
        return all_Z_log_ops_new, all_X_log_ops_new

    def find_new_teleportation_meas(self, new_Z_log_ops, new_X_log_ops,
                                    old_Z_log_ops, old_X_log_ops):
        """
        Given a new set of logical operators and old one, find which new valid teleportation measurements are generated.
        """

        if np.any(old_Z_log_ops) or np.any(old_X_log_ops):
            # print("Doing proper combinations", list(powerset(range(self.num_logical_qbits))))
            which_qubits_get_new_XZ = combinations_with_replacement(powerset(range(self.num_logical_qbits)), 2)
        else:
            which_qubits_get_new_XZ = [(tuple(range(self.num_logical_qbits)), tuple(range(self.num_logical_qbits)))]

        new_valid_tele_meas = []
        for this_newold_comb in which_qubits_get_new_XZ:
            # exclude the combinations where there are no new X and Z operators, as these have already been all counted
            # when exploring the old operators.

            if this_newold_comb != ((), ()):
                # print("in", len(old_Z_log_ops), len(old_X_log_ops))
                logops_z = [old_Z_log_ops[i] if i in this_newold_comb[0] else new_Z_log_ops[i]
                            for i in range(self.num_logical_qbits)]
                logops_x = [old_X_log_ops[i] if i in this_newold_comb[1] else new_X_log_ops[i]
                            for i in range(self.num_logical_qbits)]
                # print("out", len(old_Z_log_ops), len(old_X_log_ops))

                z_logops_combs = product(*logops_z)
                x_logops_combs = product(*logops_x)
                all_logops_combs = product(z_logops_combs, x_logops_combs)

                # z_logops_combs = list(product(*logops_z))
                # x_logops_combs = list(product(*logops_x))
                # all_logops_combs = list(product(z_logops_combs, x_logops_combs))
                # print(z_logops_combs, x_logops_combs, all_logops_combs)

                new_valid_tele_meas = new_valid_tele_meas + [y[1].op for y in
                                                             (self.verify_teleportation_condition(comb) for comb in
                                                              all_logops_combs)
                                                             if y[0]]
                # print(new_valid_tele_meas)

        return list(set(new_valid_tele_meas))

    def SPalgorithm_valid_teleportation_meas(self, max_m_increase=1., test_inouts=True, exclude_input_ys=True,
                                             return_evolution=False):

        adj_mat = nx.adjacency_matrix(self.code_graph, nodelist=sorted(self.code_graph.nodes())).todense()
        all_min_m = [find_min_cardinality_connecting_nodes(this_input_set, self.output_nodes, adj_mat,
                                                           max_iterations=None)[0] for this_input_set in
                     self.res_inputs]
        min_m = max(all_min_m)
        max_m = self.nqubits

        valid_Z_logops = [[] for i in range(self.num_logical_qbits)]
        valid_X_logops = [[] for i in range(self.num_logical_qbits)]

        teleport_meas = []

        m = min_m
        end_m = max_m
        check_notfound_combs = True
        while m < end_m:
            new_Z_logops, new_X_logops = self.find_new_logops_for_node_cardinality(m, adj_mat, test_inouts=test_inouts,
                                                                                   exclude_input_ys=exclude_input_ys)
            new_meas = self.find_new_teleportation_meas(new_Z_logops, new_X_logops, valid_Z_logops, valid_X_logops)

            # if list is nonempty, set
            if len(new_meas) > 0 and check_notfound_combs:
                check_notfound_combs = False
                end_m = m + max_m_increase
                print("Found valid measures at m=", m, "; Continuing up to end_m=", end_m)

            if return_evolution:
                teleport_meas.append((m, new_meas))
            else:
                teleport_meas = list(set(teleport_meas + new_meas))

            for i in range(self.num_logical_qbits):
                valid_Z_logops[i] = valid_Z_logops[i] + new_Z_logops[i]
                valid_X_logops[i] = valid_X_logops[i] + new_X_logops[i]

            m = m + 1

        return teleport_meas

    ##################################################
    #### TELEPORTATION - HERALDED LOSS TOLERANCE  ####
    ##################################################

    def meas_weight(self, tele_meas):
        return len(tele_meas) - tele_meas[:self.res_graph_num_nodes].count('I')

    def allowed_lost_qubits(self, tele_meas):
        if not isinstance(tele_meas, str):
            raise ValueError("Teleportation measurement needs to be str, like 'IZIZZXYXZ'")
        meas_on_res_nodes = tele_meas[:self.res_graph_num_nodes]
        return tuple([pos for pos, char in enumerate(meas_on_res_nodes) if char == 'I'])

    def Heralded_loss_teleport_prob_MC_estimation(self, teleport_meas, loss_probs, MC_trials=10000,
                                                  use_indip_loss_combs=True):

        all_loss_combs = list(set(map(self.allowed_lost_qubits, teleport_meas)))
        if use_indip_loss_combs:
            loss_combs_to_use = get_independent_loss_combs(all_loss_combs)
        else:
            loss_combs_to_use = all_loss_combs
        if isinstance(loss_probs, (list, Iterable)):
            tele_prob_list = []
            for this_loss_prob in loss_probs:
                succ_loss_corr = 0
                for trial_ix in range(MC_trials):
                    lost_nodes = set([i for i in self.res_graph_nodes if np.random.rand() < this_loss_prob])
                    for loss_comb in loss_combs_to_use:
                        if lost_nodes.issubset(set(loss_comb)):
                            succ_loss_corr = succ_loss_corr + 1
                            break
                tele_prob_list.append(succ_loss_corr / MC_trials)
            return tele_prob_list
        else:
            succ_loss_corr = 0
            for trial_ix in range(MC_trials):
                lost_nodes = set([i for i in self.res_graph_nodes if np.random.rand() < loss_probs])
                for loss_comb in loss_combs_to_use:
                    if lost_nodes.issubset(set(loss_comb)):
                        succ_loss_corr = succ_loss_corr + 1
                        break
            return succ_loss_corr / MC_trials

    ####################################################
    #### TELEPORTATION - UNHERALDED LOSS TOLERANCE  ####
    ####################################################

    @staticmethod
    def conditional_meas_sublist(meas_list, cond_qubit_meas, measured_qubit=0):
        return [meas[0:measured_qubit:] + meas[measured_qubit + 1::] for meas in meas_list
                if (meas[measured_qubit] == 'I' or meas[measured_qubit] == cond_qubit_meas)]

    def depthzero_best_meas(self, meas_list, weight_fact, loss_prob):
        """
        This is the algorithm for choosing the measurement strategy with unheralded losses from Morley-Short's paper,
        where no Bayesian optimization is performed. A factor weight_fact is also inserted to chose how to pick the
        measurement dependening on the measurement weights: if weight_fact=0 we get the 'most-common' measurement
        strategy, if weight_fact>>1 we get the 'max-tolerance' strategy, values in between are tradeoffs between
        these situations.
        """
        ops_weighted_counts = [0, 0, 0, 0]
        for meas in meas_list:
            op_ix = PauliOps.index(meas[0])
            meas_weight = self.meas_weight(meas)
            if meas_weight == 0:
                return 'I', 1
            ops_weighted_counts[op_ix] = ops_weighted_counts[op_ix] + ((1 / meas_weight) ** weight_fact)
        best_op = PauliOps[ops_weighted_counts.index(max(ops_weighted_counts))]

        if best_op == 'I':
            # print('SMS algo returning', best_op, 1)
            return best_op, 1
        else:
            # print('SMS algo returning', best_op, 1-loss_prob)
            return best_op, 1 - loss_prob

    def get_UnheraldLT_opt_meas(self, meas_list, loss_prob, weight_fact=4, tree_depth=0, max_tree_depth=0):
        """
        Function to get the best Pauli measurement strategy, based on on a given
        """
        # if there are no measurements left, return empty string and 0 probability.
        # print('Starting test with tree_depth', tree_depth)
        if len(meas_list) == 0:
            # print('NO MEASUREMENT TO USE')
            return '', 0
        # If there is only a last qubit left, or if we are at the max depth of the decision tree, do standard strategy
        # to determine the measurement for this single qubit.
        elif (len(meas_list[0]) == 1) or (tree_depth == max_tree_depth):
            # print('DOING SMS: Reached last qubit or max_tree_depth:', meas_list[0], tree_depth)
            return self.depthzero_best_meas(meas_list, weight_fact, loss_prob)
        # otherwise, adopt more general strategy
        else:
            meas_strategies = [self.get_UnheraldLT_opt_meas(self.conditional_meas_sublist(meas_list, single_meas),
                                                            loss_prob, weight_fact=weight_fact,
                                                            tree_depth=(tree_depth + 1), max_tree_depth=max_tree_depth)
                               for single_meas in PauliOps]
            # print('meas_strategies:', meas_strategies)
            strat_prob = [x[1] for x in meas_strategies]
            best_prob = max(strat_prob)
            best_prob_ix = strat_prob.index(best_prob)
            best_strat = meas_strategies[best_prob_ix][0]
            best_meas = PauliOps[best_prob_ix]
            # print('choosing strat: ', best_strat)

            if best_meas == 'I':
                return best_meas + best_strat, best_prob
            else:
                return best_meas + best_strat, best_prob * (1 - loss_prob)

    def Unheralded_loss_teleport_single_MC_trial(self, teleport_meas, loss_prob, follow_curr_best=False,
                                                 weight_fact=4, max_tree_depth=0):

        current_meas_list = [meas[:self.res_graph_num_nodes] for meas in teleport_meas]
        current_best_strategy = ''
        check_measured = True
        for this_node in list(self.res_graph_nodes):
            # print("\nNEW START: starting node", this_node, 'with current strat:', current_best_strategy, '\nand current meas_list',current_meas_list)
            # print('Current loss_prob', loss_prob)
            # Check if there are still some possible measurement strategies left. If not, we cannot perform
            # teleportation and 0 is returned.
            if not current_meas_list:
                return 0

            # Decide the best measurement to do
            if follow_curr_best and (len(current_best_strategy) > 0):
                tried_meas = current_best_strategy[0]
            else:
                current_best_strategy, temp_prob = self.get_UnheraldLT_opt_meas(current_meas_list, loss_prob,
                                                                                weight_fact=weight_fact,
                                                                                max_tree_depth=max_tree_depth)
                # Check if current_best_strategy exists. If not, we cannot perform teleportation and 0 is returned.
                # print('Obtained best strategy:', current_best_strategy, temp_prob)
                if not current_best_strategy:
                    return 0
                else:
                    tried_meas = current_best_strategy[0]
                    # print('Measurement on this qubit:', tried_meas)

            # determine if a qubit is lost with probability loss_prob
            if np.random.rand() < loss_prob:
                is_lost = True
                # print('Photon is LOST')
            else:
                is_lost = False
                # print('Photon is ALIVE!!')

            if is_lost and (tried_meas != 'I'):
                # print('MEASUREMENT NOT PERFORMED!!')
                check_measured = False
                # update measurement list
                current_meas_list = self.conditional_meas_sublist(current_meas_list, 'I')
                # update current best measurement strategy
                current_best_strategy = ''
            else:
                # print('MEASUREMENT ACHIEVED!!')
                check_measured = True
                # update measurement list
                current_meas_list = self.conditional_meas_sublist(current_meas_list, tried_meas)
                # update current best measurement strategy
                current_best_strategy = current_best_strategy[1:]

        # if we manage to conclude the "for" cycle over all nodes, we managed to get a strategy for succesfull
        # teleportation, and we return 1.
        if check_measured:
            # print('\n\nTELEPORTATION DONE!')
            return 1
        else:
            return 0

    def Unheralded_loss_teleport_prob_MC_estimation(self, meas_list, loss_probs, weight_fact=4,
                                                    follow_curr_best=False, max_tree_depth=0,
                                                    MC_trials=10000):

        if isinstance(loss_probs, (list, Iterable)):
            tele_prob_list = []
            for this_loss_prob in loss_probs:
                succ_loss_corr = 0
                for trial_ix in range(MC_trials):
                    succ_loss_corr = succ_loss_corr \
                                     + self.Unheralded_loss_teleport_single_MC_trial(meas_list, this_loss_prob,
                                                                                     follow_curr_best=follow_curr_best,
                                                                                     weight_fact=weight_fact,
                                                                                     max_tree_depth=max_tree_depth)
                tele_prob_list.append(succ_loss_corr / MC_trials)
            return tele_prob_list
        else:
            succ_loss_corr = 0
            for trial_ix in range(MC_trials):
                succ_loss_corr = succ_loss_corr \
                                 + self.Unheralded_loss_teleport_single_MC_trial(meas_list, loss_probs,
                                                                                 follow_curr_best=follow_curr_best,
                                                                                 weight_fact=weight_fact,
                                                                                 max_tree_depth=max_tree_depth)
            return succ_loss_corr / MC_trials

    #####################################################################
    #### ALGORITHM FOR NEW TYPES OF CODES WITH NO FIXED OUTPUT NODES ####
    #####################################################################

    # TODO: THINK ABOUT MAKING THE ALGORITHM WORK FOR MORE THAN 1 ENCODED QUBIT.

    def all_logops_uptocardinality(self, max_m=5.):
        # stabnodes = powerset_nonempty_tomax(list(self.res_graph_nodes) + list(self.output_nodes)
        #                                     , min([self.nqubits, max_m]))
        stabnodes = list(powerset_nonempty_tomax(list(self.res_graph_nodes) + list(self.output_nodes)
                                                 , min([self.nqubits, max_m])))
        # For all possible combinations of nodes multiplity the associated stab generators to get
        # the associated stabilizers
        labelles_stab_gens = self.get_code_stabilizer_gens_labelled()
        allstabs = [reduce(lambda x, y: x * y, [labelles_stab_gens[ix] for ix in this_nontriv_stabnodes]) for
                    this_nontriv_stabnodes in stabnodes]

        [Z_log_ops, X_log_ops] = self.logical_ops
        all_Z_log_ops = [[(this_stab * this_log_op).op for this_stab in allstabs] for this_log_op in Z_log_ops]
        # Keep only the X ones with condition that on the input qubit they have 'I' (otherwise they won't commute
        # with the Z logops on the input qubit)
        all_X_log_ops = [[y for y in ((this_stab * this_log_op).op for this_stab in allstabs)
                          if y[self.input_nodes[0]] == 'I']
                         for this_log_op in X_log_ops]
        return all_Z_log_ops[0], all_X_log_ops[0]

    def max_tolerance_meas_ULTRA(self, qubits_left, z_logops, x_logops):
        ops_weighted_counts = [[0, 0, 0, 0] for i in range(len(qubits_left))]

        # get z_logops with minimum weight
        z_weight_list = list(map(self.meas_weight, z_logops))
        min_z_weight = min(z_weight_list)
        minweight_z_logops = [logop for ix, logop in enumerate(z_logops) if z_weight_list[ix] == min_z_weight]

        # get x_logops with minimum weight
        x_weight_list = list(map(self.meas_weight, x_logops))
        min_x_weight = min(x_weight_list)
        minweight_x_logops = [logop for ix, logop in enumerate(x_logops) if x_weight_list[ix] == min_x_weight]

        max_min_weigths = max([min_z_weight, min_x_weight])

        # get measurement comparing only the minimum weight logops
        for this_z_logop in minweight_z_logops:
            for this_x_logop in minweight_x_logops:
                for qbt_ix in range(len(qubits_left)):
                    z_op = this_z_logop[qbt_ix]
                    x_op = this_x_logop[qbt_ix]
                    if z_op == 'I':
                        op_ix = PauliOps.index(x_op)
                        ops_weighted_counts[qbt_ix][op_ix] = ops_weighted_counts[qbt_ix][op_ix] + 1
                    elif (z_op == x_op) or (x_op == 'I'):
                        op_ix = PauliOps.index(z_op)
                        ops_weighted_counts[qbt_ix][op_ix] = ops_weighted_counts[qbt_ix][op_ix] + 1

        # Pick the best qubit to measure, i.e. the one which has a valid-measurement operator appearing in most valid
        # x_logop - z_logop combinations, and the associated measurement operator.
        max_all_qubits_list = list(map(max, ops_weighted_counts))
        max_all_qubits = max(max_all_qubits_list)
        chosen_qubit_ix = max_all_qubits_list.index(max_all_qubits)
        if max_all_qubits == 0:
            chosen_meas_op = 'I'
        else:
            chosen_meas_op = PauliOps[ops_weighted_counts[chosen_qubit_ix].index(max_all_qubits)]
        chosen_qubit = qubits_left[chosen_qubit_ix]
        return chosen_qubit, chosen_meas_op, max_min_weigths

    def ULTRA_single_MC_trial(self, loss_prob, max_m=5.):
        all_z_logops_list, all_x_logops_list = self.all_logops_uptocardinality(max_m=max_m)
        print(all_z_logops_list)
        # pick only the operators on the nodes in res_graph (we assume the input qubit exists and is measured in the
        # X or Y basis associated to the Zlogop used in the final measurement strategy).
        z_logops = [this_logop[:self.res_graph_num_nodes] for this_logop in all_z_logops_list]
        x_logops = [this_logop[:self.res_graph_num_nodes] for this_logop in all_x_logops_list]

        qubits0 = list(self.res_graph_nodes)
        qubits_left = qubits0

        for step in range(len(qubits0)):
            print('Starting step', step, 'with qubits', qubits_left)
            print('z_logops', z_logops)
            print('x_logops', x_logops)

            is_lost = False

            # if any of the available logical operators or available qubits lists areempty, we failed.
            if (not z_logops) or (not x_logops) or (not qubits_left):
                return 0

            chosen_qubit, chosen_meas_op, max_min_weight = self.max_tolerance_meas_ULTRA(qubits_left, z_logops, x_logops)
            print('Obtained optimal measurement: qubit', chosen_qubit, '; op',chosen_meas_op)

            # if there exist both x and z logical operators with weight <= 1, check if they can provide teleportation.
            if max_min_weight <= 1:
                print('\nCOOL! Got max_min_weight',max_min_weight,'...Trying teleportation')
                weight1_z_logops = [logop for logop in z_logops if (self.meas_weight(logop) == 1)]
                weight1_x_logops = [logop for logop in x_logops if (self.meas_weight(logop) == 1)]
                z1_logops_elems = [next((ix, x) for ix, x in enumerate(this_logop) if x != 'I') for this_logop
                                   in weight1_z_logops]
                x1_logops_elems = [next((ix, x) for ix, x in enumerate(this_logop) if x != 'I') for this_logop
                                   in weight1_x_logops]
                print(z1_logops_elems)
                print(x1_logops_elems)
                for (z_qubit_ix, z_qubit_op) in z1_logops_elems:
                    # if a teleportation try wasn't already in this step, keep going
                    if not is_lost:
                        for (x_qubit_ix, x_qubit_op) in x1_logops_elems:
                            # if they act on the same qubit, keep going
                            print((z_qubit_ix, z_qubit_op), (x_qubit_ix, x_qubit_op))
                            if z_qubit_ix == x_qubit_ix:
                                print('Acting on same qubit!')
                                # if the associated operators anticommute, keep going
                                if not (z_qubit_op == x_qubit_op or z_qubit_op == 'I' or x_qubit_op == 'I'):
                                    print('Anticommute!')
                                    # try to perform teleportation using this qubit
                                    tried_teleport = True

                                    # determine with probability loss_prob if photon is lost. If it is not lost,
                                    # we successfully achieved teleportation.
                                    if np.random.rand() > loss_prob:
                                        print('PHOTON FOUND! TELEPORTATION SUCCESSFUL!')
                                        return 1
                                    else:
                                        print('Photon is lost, keep trying\n')
                                        # if photon is lost, remove the qubits from the available ones, and update
                                        # logical operators.
                                        is_lost = True
                                        qubits_left.remove(z_qubit_ix)
                                        # TODO: check if this way of updating logical operators is correct
                                        z_logops = self.conditional_meas_sublist(z_logops, 'I', z_qubit_ix)
                                        x_logops = self.conditional_meas_sublist(x_logops, 'I', z_qubit_ix)

            # if there are no z_logops and x_logops combinations with weights < 1, measure a qubit to reduce weights
            if not is_lost:

                # determine if a qubit is lost with probability loss_prob
                if np.random.rand() < loss_prob:
                    is_lost = True
                    # print('Photon is LOST')
                else:
                    is_lost = False
                    # print('Photon is ALIVE!!')

                if is_lost:
                    # TODO: check if this way of updating logical operators is correct
                    # update logical operators
                    chosen_qubit_ix = qubits_left.index(chosen_qubit)
                    z_logops = self.conditional_meas_sublist(z_logops, 'I', chosen_qubit_ix)
                    x_logops = self.conditional_meas_sublist(x_logops, 'I', chosen_qubit_ix)
                    # Remove qubit from available list
                    qubits_left.remove(chosen_qubit)
                else:
                    # TODO: check if this way of updating logical operators is correct
                    # update logical operators
                    chosen_qubit_ix = qubits_left.index(chosen_qubit)
                    z_logops = self.conditional_meas_sublist(z_logops, chosen_meas_op, chosen_qubit_ix)
                    x_logops = self.conditional_meas_sublist(x_logops, chosen_meas_op, chosen_qubit_ix)
                    # Remove qubit from available list
                    qubits_left.remove(chosen_qubit)

        # if we finish all available qubits and still didn't achieve teleportation, we failed
        return 0

    def ULTRA_teleport_prob_MC_estimation(self, loss_probs, max_m=5., MC_trials=10000):
        if isinstance(loss_probs, (list, Iterable)):
            tele_prob_list = []
            for this_loss_prob in loss_probs:
                succ_loss_corr = 0
                for trial_ix in range(MC_trials):
                    succ_loss_corr = succ_loss_corr + self.ULTRA_single_MC_trial(this_loss_prob, max_m)
                tele_prob_list.append(succ_loss_corr / MC_trials)
            return tele_prob_list
        else:
            succ_loss_corr = 0
            for trial_ix in range(MC_trials):
                succ_loss_corr = succ_loss_corr + self.ULTRA_single_MC_trial(loss_probs, max_m)
            return succ_loss_corr / MC_trials

##############################
### OTHER USEFUL FUNCTIONS ###
##############################

def find_min_cardinality_connecting_nodes(input_nodes, output_nodes, adj_mat, max_iterations=None):
    """
    Algorithm that find the minimum cardinality for the set of nodes such that the associated neighbourhood connects the
    input_nodes to output_nodes, i.e. that can produce nontrivial stabilizers.

    :return: minimum cardinality "m", and "neighbourhood" of possible nodes that con connect inputs and outputs in m
    steps.
    """
    # if max_iterations is defined and is an integer, define the maximum number steps to find a stabilizer path from
    # input nodes to output nodes as max_iterations, otherwise set it as the total number of nodes in the graph.
    nqubits = len(adj_mat)
    if not isinstance(max_iterations, int):
        max_steps = nqubits
    else:
        max_steps = max_iterations

    # initialise explored_nodes_array as the input nodes.
    explored_nodes = np.zeros(nqubits, dtype=int)
    for node_ix in input_nodes:
        explored_nodes[node_ix] = 1

    # initialise step counter iter_step, and check if explored_nodes already overlaps with the output nodes.
    iter_step = 1
    if explored_nodes[output_nodes].any():
        test_nonoverlap = False
    else:
        test_nonoverlap = True

    # do a loop where at each step explored_nodes includes all the neighbours of the previous set of nodes, and
    # check if now it overlaps with the output nodes.
    # explored_nodes = explored_nodes.T
    while test_nonoverlap and (iter_step < max_steps):
        iter_step = iter_step + 1
        # neighbourhood includes the stab_nodes plus all adjacent nodes
        explored_nodes = ((explored_nodes @ adj_mat) + explored_nodes) > 0
        if explored_nodes[0, output_nodes].any():
            test_nonoverlap = False

    neighbourhood = list(np.nonzero(explored_nodes)[1])
    m = int(iter_step / 2.)

    return m, neighbourhood


def get_independent_loss_combs(all_loss_combs):
    """
    Function that returns all the maximally loss-tolerant combinations of lost nodes in all_loss_combs:
    any of them is not a subset of any other combination, and all combinations in all_loss_combs are subsets of some
    combination in the list returned.
    """
    ind_loss_combs = []
    for ix, loss_comb in enumerate(all_loss_combs):
        check_non_incl = True
        # check if loss_comb is already a subset of a set in ind_loss_combs
        for checked_loss_comb in ind_loss_combs:
            if set(loss_comb).issubset(set(checked_loss_comb)):
                check_non_incl = False
                break
        if check_non_incl:
            # if not already in ind_loss_combs, check in the remaining combs in all_loss_combs
            for checked_loss_comb in all_loss_combs[(ix + 1):]:
                if set(loss_comb).issubset(set(checked_loss_comb)):
                    check_non_incl = False
                    break
        # if not found anywhere, add loss_comb to ind_loss_combs
        if check_non_incl:
            ind_loss_combs.append(loss_comb)
    return ind_loss_combs


def single_qubit_commute(pauli1, pauli2, qbt):
    """
    Returns 0 if the operators on the qbt-th qubit of the two operators in the Pauli group commute,
    and 1 if they anticommute.
    """
    if not (isinstance(pauli1, q.Pauli) and isinstance(pauli1, q.Pauli)):
        raise ValueError("Pauli elements need to be qecc.Pauli objects.")
    op1 = pauli1.op[qbt]
    op2 = pauli2.op[qbt]
    if op1 == 'I' or op2 == 'I' or op1 == op2:
        return 0
    else:
        return 1


def single_qubit_commute_on_qubit_list(pauli1, pauli2, qbt_list):
    return all([single_qubit_commute(pauli1, pauli2, qbt_idx) == 0 for qbt_idx in qbt_list])


def rotate(point, angle, origin=(0, 0)):
    """
    Rotate a point counterclockwise by a given angle (in radiants) around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def powerset_nonempty(iterable):
    """powerset_nonempty([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def powerset_nonempty_tomax(iterable, k):
    """powerset_nonempty([1,2,3], 2) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, k + 1))


##############################
######       TESTS      ######
##############################


if __name__ == '__main__':
    from CodesFunctions.graphs import *
    import time

    ########## Crazy-graph encoding
    nrows = 4
    nlayers = 1
    encode_graph = gen_crazy_graph(nrows, nlayers)
    # encode_graph = gen_square_lattice_graph(nrows, nlayers)
    # encode_graph = gen_triangular_lattice_graph(nrows, nlayers)
    # encode_graph = gen_hexagonal_lattice_graph(nrows, nlayers)
    # encode_graph = gen_multiwire_graph(nrows, nlayers)
    in_nodes = list(range(nrows))
    # out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))

    # in_nodes = [[0, 1, 2], [3, 4, 5]]
    # out_nodes = [[6, 7, 8], [9, 10]]
    # in_nodes = [[0, 1]]
    # out_nodes = [[0, 1]]
    # in_nodes = [[0, 1], [2]]
    # out_nodes = [[0, 1], [2]]
    # in_nodes = [[0, 1], [2, 3]]
    # out_nodes = [[4, 5], [6, 7]]
    out_nodes = []

    ########## gen_fullyconnected_graph
    # nqbts = 16
    # encode_graph = gen_linear_graph(nqbts)
    # encode_graph = gen_fullyconnected_graph(nqbts)
    # encode_graph = gen_ring_graph(nqbts)
    # in_nodes = list(range(int(nqbts / 2)))
    # out_nodes = list(range(int(nqbts / 2), nqbts))

    ##################
    ### START TEST ###
    ##################

    mycode = LTCode(encode_graph, in_nodes, out_nodes)

    # qwe = mycode.full_allowed_heralded_loss_combinations(trivial_stab_test=False, exclude_input_ys=True)

    # start_time = time.time()
    # qwe = mycode.full_search_valid_teleportation_meas(trivial_stab_test=False, exclude_input_ys=True)
    # qwe = mycode.SPalgorithm_valid_teleportation_meas(max_m_increase_fact=2, test_inouts=True,
    #                                                   exclude_input_ys=True, return_evolution=False)
    # end_time = time.time()
    # print()
    # print("Finished function, took:", end_time - start_time, "s")
    # print(len(qwe))
    # print(qwe)

    ############### NEW TESTS ADAPTIVE LOSS TOLERANCE

    # asd = mycode.all_logops_uptocardinality(3)
    loss_prob = 0
    asd = mycode.ULTRA_single_MC_trial(loss_prob, max_m=5.)
    print(asd)

    ############### PRINT

    plt.figure()
    mycode.image(with_labels=True)
    plt.show()
