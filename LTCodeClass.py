from GraphStateClass import GraphState
# from StabStateClass import StabState

import qecc as q

from copy import deepcopy
import numpy as np
from itertools import product


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
        if len(input_vertices) == 0 or len(output_vertices) == 0:
            raise ValueError("At least one qubit and one input/output mode associated to it are needed")

        if isinstance(input_vertices[0], int):
            in_vtx = [input_vertices]
        else:
            in_vtx = input_vertices

        if isinstance(output_vertices[0], int):
            out_vtx = [output_vertices]
        else:
            out_vtx = output_vertices

        if len(in_vtx) != len(out_vtx):
            raise ValueError("Number of input qubits needs to be the same as number of output qubits")
        self.num_logical_qbits = len(in_vtx)

        self.res_graph = encoding_graph

        self.res_inputs = in_vtx
        self.res_outputs = out_vtx

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
        self.nqubits = self.res_graph_num_nodes + 2 * self.num_logical_qbits
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
            range(self.res_graph_num_nodes + self.num_logical_qbits, self.res_graph_num_nodes +
                  2 * self.num_logical_qbits)
        )
        out_edges = []
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
        for out_idx in range(self.num_logical_qbits):
            options_nodes = {"node_size": 150, "alpha": 1, "node_color": "mediumblue"}
            nx.draw_networkx_nodes(self.code_graph, pos_nodes, nodelist=[self.output_nodes[out_idx]], **options_nodes)
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


##############################
### OTHER USEFUL FUNCTIONS ###
##############################


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


##############################
######       TESTS      ######
##############################


if __name__ == '__main__':
    from graphs import *
    import matplotlib.pyplot as plt
    import networkx as nx

    ########## Crazy-graph encoding
    nrows = 6
    nlayers = 2
    encode_graph = gen_crazy_graph(nrows, nlayers)
    # in_nodes = list(range(nrows))
    # out_nodes = list(range((nlayers - 1) * nrows, nrows * nlayers))
    in_nodes = [[0, 1, 2], [3, 4, 5]]
    out_nodes = [[6, 7, 8], [9, 10]]

    ########## gen_fullyconnected_graph
    # nqbts = 6
    # encode_graph = gen_linear_graph(nqbts)
    # encode_graph = gen_fullyconnected_graph(nqbts)
    # encode_graph = gen_ring_graph(nqbts)
    # in_nodes = list(range(int(nqbts / 2)))
    # out_nodes = list(range(int(nqbts / 2), nqbts))

    mycode = LTCode(encode_graph, in_nodes, out_nodes)

    print(mycode.logical_ops)
    print(mycode.stab_gens)

    plt.figure()
    mycode.image(with_labels=True)
    plt.show()




    resource_graph_state = GraphState(mycode.res_graph)
    total_graph_state = GraphState(mycode.code_graph)
