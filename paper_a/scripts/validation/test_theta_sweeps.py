import numpy as np
from scipy.linalg import expm

from plot_theta_sweeps import alignment
from recalculate_exact import entropy, seeded, squared_overlaps


def test_fragment_holevo_against_density_matrices():
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    times = np.array([0., 1.3, 27.])
    for p in (0., .25, .5, .75, 1.):
        params = dict(nqubits_E=2, H_SE_J=.1, Mironowicz_h0=.08,
                      Mironowicz_alpha2=.13, Mironowicz_theta=.7, psi_bias=p)
        seed = 3923535749
        rg, rh = seeded(seed, 'mironowicz.J_SE'), seeded(seed, 'mironowicz.J_E')
        gs = [rg.gauss(0, .1) for _ in range(2)]
        hs = [.08+rh.gauss(0, .13) for _ in range(2)]
        b2 = squared_overlaps(params, seed, times)
        for it, t in enumerate(times):
            branches = [np.array([1.], complex), np.array([1.], complex)]
            for g, h in zip(gs, hs):
                initial = np.array([np.sqrt(p), np.sqrt(1-p)])
                h0 = h*z
                h1 = h*z + np.pi*g/2*(np.eye(2)-np.cos(.7)*x-np.sin(.7)*z)
                for j, hamiltonian in enumerate((h0, h1)):
                    branches[j] = np.kron(branches[j], expm(-1j*t*hamiltonian)@initial)
            rho = sum(np.outer(v, v.conj()) for v in branches)/2
            eig = np.linalg.eigvalsh(rho)
            eig = eig[eig > 1e-14]
            direct = -np.sum(eig*np.log2(eig))
            predicted = entropy(np.sqrt(np.prod(b2[:, it])))/np.log(2)
            np.testing.assert_allclose(predicted, direct, atol=2e-13)


def test_no_field_preparation_ordering_and_peak():
    times = np.arange(20., 41.)
    params = dict(nqubits_E=16, H_SE_J=.1, Mironowicz_h0=0., Mironowicz_alpha2=0., psi_bias=.25)
    a = squared_overlaps(dict(params, Mironowicz_theta=0.), 3212275105, times)
    b = squared_overlaps(dict(params, Mironowicz_theta=np.pi/3), 3212275105, times)
    c = squared_overlaps(dict(params, Mironowicz_theta=np.pi/2), 3212275105, times)
    assert np.all(b <= c + 2e-14)
    assert np.all(c <= a + 2e-14)
    np.testing.assert_allclose(alignment(.25, np.array([0, np.pi/3, np.pi/2])), [.25, 1, .75], atol=2e-14)


def test_commuting_endpoint_is_field_independent():
    for p in (0., .25, .5):
        params = dict(nqubits_E=16, H_SE_J=.1, Mironowicz_theta=np.pi/2, psi_bias=p)
        times = np.arange(20., 41.)
        reference = squared_overlaps(dict(params, Mironowicz_h0=0., Mironowicz_alpha2=0.), 4130349263, times)
        for h0, width in ((1., 0.), (0., 1.), (.5, .3)):
            actual = squared_overlaps(dict(params, Mironowicz_h0=h0, Mironowicz_alpha2=width), 4130349263, times)
            np.testing.assert_allclose(actual, reference, atol=5e-14)
