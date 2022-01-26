import numpy as np
from qutip import *


def get_Omega0(Topt_tilde):
    return np.sqrt(0.5 * np.pi) / (Topt_tilde * 0.96610514647)


def get_Omega(t, args):
    if isinstance(args, dict):
        return get_Omega0(args['Topt_tilde']) * np.exp(- (t ** 2) / (2 * (args['Topt_tilde'] ** 2)))
    else:
        return get_Omega0(args) * np.exp(- (t ** 2) / (2 * (args ** 2)))

# def evolve_excitation(in_state, )

##################################################
####### Differential equation solving systems ####
##################################################
#
# ### state evolution, also for time dependent Hamiltonians
# def evolve_full_excitation(t, t0, in_state, w_g, Delta_l, w_t, Topt_tilde, n_steps=2):
#     H_func = lambda y, x: get_bloch_hamiltonian(x, w_g, Delta_l, w_t, Topt_tilde)
#     return odeint(H_func, in_state, np.linspace(t0, t0+t, n_steps))
#
# def evolve_excitation_state(t, t0, in_state, w_g, Delta_l, w_t, Topt_tilde):
#     return evolve_full_excitation(t, t0, in_state, w_g, Delta_l, w_t, Topt_tilde, n_steps=2)[1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ######################### Parameters definitions

    w_g = 2 * np.pi * 17 / 2.  # in GHz
    w_t = w_g
    Delta_l = 0  # in GHz

    gammax = 25.4  # in GHz
    gammay = 0  # in GHz
    gamma0 = gammax + gammay

    Topt = 0.0356  # in ns
    Topt_tilde = Topt / (2 * np.sqrt(np.log(2)))

    ######################### Simulation definitions

    evol_time = 6 * Topt
    in_state = np.array([0., 1., 0., 0.])
    in_state = Qobj(in_state)

    num_steps = 100
    t_list = np.linspace(-evol_time / 2., evol_time / 2., num_steps)


    H0 = Qobj(np.array([[-w_g, 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., Delta_l - 0.5j*gamma0, 0.],
                        [0., 0., 0., Delta_l + w_t - 0.5j*gamma0]]))

    H1 = Qobj(0.5 * np.eye(4)[::-1])

    # H = [H0, [H1, omega_t_func]]
    H = [H0, [H1, get_Omega]]

    # output = mesolve(H, in_state, t_list)
    output = mesolve(H, in_state, t_list, args={'Topt_tilde': Topt_tilde})

    print(output)
    # print(output.states)

    vals_0 = [np.abs(state[0][0, 0]) ** 2 for state in output.states]
    vals_1 = [np.abs(state[1][0, 0]) ** 2 for state in output.states]
    vals_2 = [np.abs(state[2][0, 0]) ** 2 for state in output.states]
    vals_3 = [np.abs(state[3][0, 0]) ** 2 for state in output.states]
    inn_prods = [np.abs((state.dag() * state)[0, 0]) for state in output.states]
    print('inn_prods', inn_prods)
    omega_t_vals = [get_Omega(t, Topt_tilde) / get_Omega0(Topt_tilde) for t in t_list]

    print(vals_0)

    plt.plot(t_list, vals_0, label='0')
    plt.plot(t_list, vals_1, label='1')
    plt.plot(t_list, vals_2, label='2')
    plt.plot(t_list, vals_3, label='3')
    # plt.plot(t_list, inn_prods, label=r'$\langle \psi | \psi \rangle$')
    plt.plot(t_list, omega_t_vals, label='omega')
    plt.legend()
    plt.show()

    # data = evolve_full_excitation(evol_time, evol_time/2., in_state, w_g, Delta_l, w_t, Topt_tilde, n_steps=2)
    # print(data)
