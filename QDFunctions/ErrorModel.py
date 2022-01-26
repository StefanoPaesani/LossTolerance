import numpy as np
from scipy.linalg import block_diag

pauli_x = np.array([[0., 1.], [1., 0.]])
pauli_y = np.array([[0., -1.j], [1.j, 0.]])
pauli_z = np.array([[1., 0.], [0., -1.]])


###########################################
###########  Errors Functions  ############
###########################################

def state_initialization(fid, n_qubits):
    if 0 < np.random.rand() < fid:
        return multi_kron_prod(
            [np.array([1, 0, 0, 0]) if i == 0 else np.array([1, 0]) for i in range(4 * n_qubits + 1)])
    else:
        return multi_kron_prod(
            [np.array([0, 1, 0, 0]) if i == 0 else np.array([1, 0]) for i in range(4 * n_qubits + 1)])


def init_OH_field(T2star):
    if T2star < 0:
        raise ValueError('T2star needs to be > 0')
    return np.random.normal(0., np.sqrt(2) / T2star)


########## Spin rotations

def get_spin_rotation_matrix(time, omega_r, delta_mw, phi_s, n_qubits):
    theta = 0.5 * time * np.sqrt(omega_r ** 2 + delta_mw ** 2)
    temp_mat = np.cos(theta) * np.identity(2) - 1.j * np.sin(theta) * \
               (omega_r * (np.cos(phi_s) * pauli_x + np.sin(phi_s) * pauli_y) - delta_mw * pauli_z) / \
               np.sqrt(omega_r ** 2 + delta_mw ** 2)
    return np.kron(block_diag(temp_mat, np.identity(2)), np.identity(2 ** (4 * n_qubits)))


def get_spin_flip_matrices(n_qubits):
    op1 = np.zeros((4, 4))
    op1[0, 1] = 1.
    op2 = np.zeros((4, 4))
    op2[1, 0] = 1.
    return [np.kron(op1, np.identity(2 ** (4 * n_qubits))),
            np.kron(op2, np.identity(2 ** (4 * n_qubits)))]


################################################
###############  Misc Functions  ###############
################################################

def multi_kron_prod(array_list):
    if len(array_list) > 0:
        temp_array = array_list[0]
        for this_arr in array_list[1:]:
            temp_array = np.kron(temp_array, this_arr)
        return temp_array
    else:
        return np.array([])


################################################
###################  Main  #####################
################################################

if __name__ == '__main__':
    n_qbts = 2
    asd = state_initialization(1, n_qbts)

    print(asd, len(asd), np.log2(len(asd)))

    t = np.pi
    asd = get_spin_rotation_matrix(t, 1, 0, 0, 0)
    print(asd)
