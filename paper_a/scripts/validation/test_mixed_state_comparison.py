import numpy as np

from mixed_state_comparison import (
    I, X, binary_entropy, branch_data, distance, full_evolution, measurement_holevo,
    optimize_sbs, sbs_state,
)


def test_perfect_cnot_produces_exact_record_and_sbs():
    rho, metrics = branch_data(0., 0., 0.)
    assert abs(metrics['chi_z'] - 1) < 1e-12
    assert metrics['fidelity2'] < 1e-12
    assert metrics['coherence'] < 1e-12
    assert distance(rho, [0, 0, 0, np.pi, 0]) < 1e-12


def test_no_interaction_sbs_can_have_no_record():
    rho, metrics = branch_data(.7, .3, 0., t=0.)
    assert abs(metrics['chi_z']) < 1e-12
    assert distance(rho, [1, np.pi/2, 0, np.pi, 0]) < 1e-12
    free, _, _ = optimize_sbs(rho, 10, 6)
    fixed, _, _ = optimize_sbs(rho, 11, 6, fixed_z=True)
    assert free < 1e-10
    # System coherence alone gives a lower bound; a pure Z candidate gives sqrt(2).
    assert 1 - 1e-8 <= fixed <= np.sqrt(2) + 1e-8


def test_maximally_mixed_environment_has_no_z_records():
    for theta, h in [(.1, .3), (.7, 1.4), (1.3, 2.7)]:
        rho, metrics = branch_data(theta, h, .5)
        assert abs(metrics['chi_z']) < 1e-12
        assert abs(metrics['fidelity2'] - 1) < 1e-12
        assert abs(measurement_holevo(rho, 0, 0)) < 1e-12


def test_direct_evolution_with_complex_coherence():
    for theta, h, p, t in [(.6, .7, .2, .37), (1.1, 2.3, .4, 2.6)]:
        rho, metrics = branch_data(theta, h, p, t)
        np.testing.assert_allclose(rho, full_evolution(theta, h, p, t), atol=1e-13)
        assert metrics['fidelity_error'] < 1e-12
        assert metrics['holevo_error'] < 1e-12


def test_sbs_parameterization_has_correct_product_endpoints():
    state = sbs_state([1, np.pi/2, 0, np.pi/2, 0])
    np.testing.assert_allclose(state, np.kron((I+X)/2, (I+X)/2), atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(sbs_state([.4, .7, .3, 1.2, 2.1])),
                               [0, 0, .3, .7], atol=1e-14)


def test_purity_monotonicity_and_conditional_holevo():
    previous = 1.
    for p in np.linspace(0, .5, 11):
        rho, metrics = branch_data(.7, 1.2, p)
        assert metrics['chi_z'] <= previous + 1e-12
        previous = metrics['chi_z']
        assert abs(measurement_holevo(rho, 0, 0) - previous) < 1e-12


def test_mixed_controlled_z_can_correlate_x_without_recording_z():
    p = .1
    rho, metrics = branch_data(np.pi/2, 0., p)
    expected = binary_entropy((1+(1-2*p)**2)/2)-binary_entropy(p)
    assert abs(metrics['chi_z']) < 1e-12
    assert abs(measurement_holevo(rho, np.pi/2, 0.)-expected) < 1e-12
    assert expected > .2


def test_four_qubit_direct_evolution_and_extra_traced_witness():
    for theta, h, p, t in [(.6, .7, .2, .37), (1.1, 2.3, .4, 2.6)]:
        old, old_metrics = branch_data(theta, h, p, t)
        rho, metrics = branch_data(theta, h, p, t, environment_qubits=3)
        np.testing.assert_allclose(rho, full_evolution(theta, h, p, t, 3), atol=1e-13)
        np.testing.assert_allclose(rho[:2, :2], old[:2, :2], atol=1e-14)
        np.testing.assert_allclose(rho[2:, 2:], old[2:, 2:], atol=1e-14)
        assert abs(metrics['chi_z']-old_metrics['chi_z']) < 1e-14
        assert abs(metrics['coherence']-old_metrics['coherence']**2) < 1e-14


def test_four_qubit_controlled_z_transverse_information():
    p = .1
    eta = 1 - 2*p
    rho, metrics = branch_data(np.pi/2, 0., p, environment_qubits=3)
    expected = binary_entropy((1+eta**3)/2)-binary_entropy((1+eta**2)/2)
    assert abs(metrics['chi_z']) < 1e-12
    assert abs(measurement_holevo(rho, np.pi/2, 0.)-expected) < 1e-12


def test_matched_population_coherent_preparations():
    for coherence in [0., .4, 1.]:
        rho, metrics = branch_data(.8, .6, .25, .7, 3, coherence)
        np.testing.assert_allclose(rho, full_evolution(.8, .6, .25, .7, 3, coherence), atol=1e-13)
        assert metrics['fidelity_error'] < 1e-12
        assert metrics['holevo_error'] < 1e-12
    for h in [0., .7]:
        pure, metrics = branch_data(.8, h, 0., environment_qubits=3, initial_coherence=1.)
        mixed, _ = branch_data(.8, h, 0., environment_qubits=3)
        np.testing.assert_allclose(pure, mixed, atol=1e-14)


def test_matched_half_population_cnot_and_cz_limits():
    _, blind = branch_data(0., 0., .5, environment_qubits=3, initial_coherence=1.)
    _, record = branch_data(np.pi/2, 0., .5, environment_qubits=3, initial_coherence=1.)
    _, mixed = branch_data(np.pi/2, 0., .5, environment_qubits=3)
    assert abs(blind['chi_z']) < 1e-12
    assert abs(record['chi_z']-1) < 1e-12
    assert abs(mixed['chi_z']) < 1e-12
