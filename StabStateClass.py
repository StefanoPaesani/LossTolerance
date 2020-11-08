import numpy as np
import qecc as q
from stabs_graph_equiv import stab_to_graph



class StabState(object):
    r"""
    Class representing a Graph state on :math:`n` qubits.

    :param stab_gens: list of stabilizer generators defining the state
    :type stab_gens: :class:`qecc.PauliList`
    """

    def __init__(self, stab_gens):
        # stab_gens represent the n stabilizer generators defining the n-qubit stabilizer state

        # TODO: various checks, including that if the set in stab_gens are stabilizer but not necessarily
        #  the set of minimal stabilizer generators, it extracts the generators (QECC has such function).

        # Check that stab_gens is the correct object
        if not isinstance(stab_gens, q.PauliList):
            raise ValueError("Input stabilizers need to be a qecc.PauliList object.")

        self.stab_gens = stab_gens
        self.nqbits = len(stab_gens[0])



    def as_binary(self):
        """
        Obtains the binary representation of the stabilizer state
        """
        nq = self.nqbits
        bin_matr = np.zeros((2*nq, len(self.stab_gens)), dtype=np.int)
        for idx_stab, this_stab in enumerate(self.stab_gens):
            for idx_q in range(nq):
                if this_stab.op[idx_q] == 'X':
                    bin_matr[idx_q+nq, idx_stab] = 1
                elif this_stab.op[idx_q] == 'Y':
                    bin_matr[idx_q, idx_stab] = 1
                    bin_matr[idx_q+nq, idx_stab] = 1
                elif this_stab.op[idx_q] == 'Z':
                    bin_matr[idx_q, idx_stab] = 1
        return bin_matr

    def as_graphstate(self):
        """
        Obtains a Graph State locally equivalent to the stabilizer state
        """
        return stab_to_graph(self)

    def stab_vecstate(self):
        # TODO The stabilizer_subspace function from QECC is really to slow to do this...takes already 10 seconds for
        #  7 qubits. Need to find a better way to do this, possible using clifford circuits and Aaronson style stuff
        #  (see comments in the analogue function in the GraphState class).
        """
        Calculates the vector state associate to the stabilizer, and returns it as a numpy array. If there are less
        stabilizers than number of qubits, it returns a basis for the stabilized subspace, i.e. the codewords of
        the stabilizer code.
        """
        return self.stab_gens.stabilizer_subspace()

if __name__ == '__main__':
    stab_gens = ["XZII", "ZXZI", "IZXZ", "IIZX"]
    gen_list = q.PauliList(stab_gens)
    stab_state = StabState(gen_list)
    print(stab_state.as_binary())
