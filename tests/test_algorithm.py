import numpy as np
import pytest
from scipy.optimize import linprog

import sca_algorithm as alg
import sca_constants as C
import sca_evaluation as ev
import sca_ica as ica
import sca_scenario as sc
import sca_sobi as sobi


def _axes_data(angles_deg, per_axis=200, noise_deg=0.5, seed=0):
    """Punkte auf Achsen im 2-D-Raum, Vorzeichen zufällig, kleine Winkelstreuung."""
    rng = np.random.default_rng(seed)
    pts, labels = [], []
    for j, a in enumerate(angles_deg):
        th = np.radians(a + noise_deg * rng.standard_normal(per_axis))
        sign = rng.choice([-1.0, 1.0], per_axis)
        pts.append(np.column_stack([np.cos(th), np.sin(th)]) * sign[:, None])
        labels += [j] * per_axis
    return np.vstack(pts), np.array(labels)


# --- Achsen-k-Means -------------------------------------------------------------------------------------------------------------------


def test_axial_kmeans_recovers_three_axes_regardless_of_sign():
    U, labels = _axes_data([10.0, 50.0, 80.0])
    r = alg.axial_kmeans(U, 3, n_restarts=5, seed=1)
    found = sorted(float(np.degrees(np.arctan2(c[1], c[0])) % 180.0) for c in r.centers)
    assert np.allclose(found, [10.0, 50.0, 80.0], atol=0.5) and r.objective > 0.9999 and r.converged
    assert np.allclose(np.linalg.norm(r.centers, axis=1), 1.0) and (r.centers.sum(axis=1) >= 0).all()             # kanonisches Vorzeichen
    for j in range(3):
        assert len(set(r.labels[labels == j])) == 1                                                                 # jede wahre Achse = genau ein Cluster


def test_axial_kmeans_is_invariant_to_flipping_signs_of_the_data():
    U, _ = _axes_data([20.0, 70.0])
    flipped = U * np.random.default_rng(3).choice([-1.0, 1.0], len(U))[:, None]
    a = alg.axial_kmeans(U, 2, n_restarts=3, seed=2)
    b = alg.axial_kmeans(flipped, 2, n_restarts=3, seed=2)
    assert np.allclose(np.sort(np.abs(a.centers[:, 0])), np.sort(np.abs(b.centers[:, 0])), atol=1e-2)


def test_axial_kmeans_is_deterministic_and_reports_restart_objectives():
    U, _ = _axes_data([15.0, 45.0, 75.0], noise_deg=2.0)
    a = alg.axial_kmeans(U, 3, n_restarts=4, seed=5)
    b = alg.axial_kmeans(U, 3, n_restarts=4, seed=5)
    assert np.array_equal(a.centers, b.centers) and len(a.restart_objectives) == 4 and a.objective == max(a.restart_objectives)


def test_single_cluster_is_the_principal_axis():
    U, _ = _axes_data([33.0], per_axis=300)
    r = alg.axial_kmeans(U, 1, n_restarts=1)
    assert abs(np.degrees(np.arctan2(r.centers[0][1], r.centers[0][0])) % 180.0 - 33.0) < 0.3


# --- Aktive Zeitpunkte, Kernpunkte, Richtungen -------------------------------------------------------------------------------------


def test_active_samples_mark_the_spikes_and_ignore_a_noise_free_silence():
    ds = ev.make_dataset(noise=0.0)
    mask = alg.active_samples(ds.X)
    assert 0.05 < mask.mean() < 0.3 and mask[ds.active.any(axis=0)].mean() > 0.6
    assert not mask[~ds.active.any(axis=0)].any() or mask[~ds.active.any(axis=0)].mean() < 0.02          # rauschfrei: ruhige Zeitpunkte werden nicht gemeldet


def test_core_samples_are_a_subset_of_the_active_samples():
    ds = ev.make_dataset()
    assert (alg.core_samples(ds.X) <= alg.active_samples(ds.X)).all() and alg.core_samples(ds.X).sum() < alg.active_samples(ds.X).sum()


def test_directions_are_unit_vectors():
    ds = ev.make_dataset(n=3)
    U = alg.directions(ds.X, alg.core_samples(ds.X))
    assert U.shape[1] == 3 and np.allclose(np.linalg.norm(U, axis=1), 1.0)


def test_core_points_avoid_the_clustering_failures_of_all_active_samples():
    """Beleg für die Begründung in core_samples: mit allen aktiven Zeitpunkten verschmilzt das Clustering benachbarte Neuronen (5 Neuronen, 2/3/5 Elektroden, 8 Datensätze: 6 von 24 Fällen), mit den Kernpunkten nie."""
    bad_plain = bad_core = 0
    for n in (2, 3, 5):
        for seed in range(100000, 100008):
            ds = ev.make_dataset(m=5, n=n, seed=seed)
            for mask, counter in ((alg.active_samples(ds.X), "plain"), (alg.core_samples(ds.X), "core")):
                centers = alg.axial_kmeans(alg.directions(ds.X, mask), 5, n_restarts=10, seed=1).centers.T
                worst = np.nanmax(ev.axis_angles(ds.A, centers))
                if worst > 30.0:
                    bad_plain += counter == "plain"
                    bad_core += counter == "core"
    assert 3 <= bad_plain <= 10 and bad_core == 0


# --- Rekonstruktion ------------------------------------------------------------------------------------------------------------------


def _sparse_instance(n=2, k=4, N=60, seed=0, overlap=0):
    rng = np.random.default_rng(seed)
    A = np.abs(rng.standard_normal((n, k))) + 0.1
    A /= np.linalg.norm(A, axis=0)
    S = np.zeros((k, N))
    for t in range(N):
        active = rng.choice(k, 2 if t < overlap else 1, replace=False)
        S[active, t] = rng.standard_normal(len(active)) * 2 + np.sign(rng.standard_normal(len(active)))
    return A, S


def _lp_solution(A, X):
    """Exakte L1-Lösung je Spalte per scipy.optimize.linprog: min sum(p + q) s.t. A (p - q) = x, p, q >= 0."""
    n, k = A.shape
    out = []
    for t in range(X.shape[1]):
        res = linprog(np.ones(2 * k), A_eq=np.hstack([A, -A]), b_eq=X[:, t], bounds=(0, None), method="highs")
        assert res.success
        out.append(res.x[:k] - res.x[k:])
    return np.array(out).T


def test_l1_reconstruction_recovers_one_sparse_vectors_and_has_a_near_optimal_l1_norm():
    A, S = _sparse_instance(n=3, k=5, N=25, seed=4)
    X = A @ S
    ours = alg.l1_reconstruct(A, X, n_iter=150, ridge=1e-12, decay=0.85)
    lp = _lp_solution(A, X)
    assert np.allclose(lp, S, atol=1e-9) and np.allclose(ours, S, atol=1e-2)
    assert np.allclose(np.abs(X - A @ ours).max(), 0.0, atol=1e-6)                                # Nebenbedingung erfüllt
    A2, S2 = _sparse_instance(n=2, k=4, N=60, seed=0)
    ours2 = alg.l1_reconstruct(A2, A2 @ S2, n_iter=150, ridge=1e-12, decay=0.85)
    assert np.abs(ours2).sum() <= 1.01 * np.abs(S2).sum() and np.median(np.abs(ours2 - S2).max(axis=0)) < 1e-3


def test_l1_reconstruction_matches_a_linear_program_solver_also_for_two_active_sources():
    """Kreuzprüfung gegen scipy.optimize.linprog (auch dort, wo die L1-Lösung nicht die wahre ist: zwei aktive Quellen mit ähnlichen Richtungen)."""
    A, S = _sparse_instance(n=3, k=4, N=30, seed=6, overlap=30)
    X = A @ S
    ours = alg.l1_reconstruct(A, X, n_iter=150, ridge=1e-12, decay=0.85)
    assert np.allclose(ours, _lp_solution(A, X), atol=1e-2)


def test_single_reconstruction_keeps_one_source_per_time_and_is_exact_for_one_sparse_vectors():
    A, S = _sparse_instance()
    R = alg.single_reconstruct(A, A @ S)
    assert ((R != 0).sum(axis=0) <= 1).all() and np.allclose(R, S, atol=1e-9)


def test_ridge_shrinks_the_solution_towards_zero():
    A, S = _sparse_instance()
    X = A @ S
    assert np.abs(alg.l1_reconstruct(A, X, ridge=1.0)).sum() < np.abs(alg.l1_reconstruct(A, X, ridge=1e-9)).sum()


# --- Gesamtverfahren ------------------------------------------------------------------------------------------------------------------


def test_fit_sca_separates_a_noise_free_underdetermined_mixture():
    ds = ev.make_dataset(n=2, noise=0.0)
    m = alg.fit_sca(ds.X, 4, "l1", seed=1)
    assert m.A_hat.shape == (2, 4) and m.sources.shape == (4, ds.X.shape[1]) and m.clustering.converged
    assert np.nanmax(ev.axis_angles(ds.A, m.A_hat)) < 3.0 and ev.matched(ds.S, m.sources)[1].min() > 0.9
    assert not m.sources[:, ~m.mask].any()                                                       # inaktive Zeitpunkte bleiben 0


def test_fit_sca_with_one_electrode_finds_nothing_and_does_not_crash():
    ds = ev.make_dataset(n=1)
    m = alg.fit_sca(ds.X, 4)
    assert not m.sources.any() and m.A_hat.shape == (1, 4) and not m.clustering.converged
    assert alg.with_reconstruction(m, ds.X, "single") is m


def test_with_reconstruction_keeps_the_clustering():
    ds = ev.make_dataset()
    a = alg.fit_sca(ds.X, 4, "l1", seed=1)
    b = alg.with_reconstruction(a, ds.X, "single")
    assert np.array_equal(a.A_hat, b.A_hat) and b.reconstruction == "single" and not np.allclose(a.sources, b.sources)
    assert np.allclose(b.sources, alg.fit_sca(ds.X, 4, "single", seed=1).sources)


def test_fit_sca_is_deterministic():
    ds = ev.make_dataset()
    a, b = alg.fit_sca(ds.X, 4, seed=3), alg.fit_sca(ds.X, 4, seed=3)
    assert np.array_equal(a.A_hat, b.A_hat) and np.array_equal(a.sources, b.sources)


# --- Vergleichsverfahren aus ica-demo/sobi-demo (eingefroren) -----------------------------------------------------------------------


def test_fastica_copy_reproduces_the_frozen_reference():
    """Eingefrorener Wert aus ica-demo (4 Neuronen, 6 Elektroden, Rauschen 0.05, Seed 7): Neuronen-Korrelation 0.981."""
    ds = ev.make_dataset(n=6)
    m = ica.fit_ica(ds.X, 4, "logcosh", "symmetric", 1)
    assert m.converged and abs(ev.matched(ds.S, m.sources)[1].mean() - 0.981) < 2e-3


def test_sobi_copy_reproduces_the_frozen_reference():
    """Eingefrorener Wert aus sobi-demo (nur Neuronen, mittlere Verzögerungen): SOBI 0.978."""
    ds = ev.make_dataset(n=6)
    m = sobi.fit_sobi(ds.X, 4, C.SOBI_LAGS)
    assert m.converged and abs(ev.matched(ds.S, m.sources)[1].mean() - 0.978) < 2e-3


# --- Szenario ----------------------------------------------------------------------------------------------------------------------------


def test_scenario_is_the_one_of_ica_demo_for_default_rate_and_spread():
    """Eingefroren aus ica-demo/sobi-demo (Seed 7, Rauschen 0.05, 6 Elektroden): dieselben Neuronen und Elektrodensignale."""
    ds = sc.make_dataset(4, 6, 0, "gauss", 1.0, 1.0, 0.05, 20000, 7)
    assert abs(np.abs(ds.S[0][:3000]).sum() - 436.61272090571026) < 1e-6
    assert abs(np.abs(ds.X[:, :2000]).sum() - ev.make_dataset(n=6).X[:, :2000].__abs__().sum()) < 1e-9


def test_rate_scale_changes_the_firing_and_the_overlap():
    slow = ev.make_dataset(rate_scale=0.5)
    fast = ev.make_dataset(rate_scale=4.0)
    assert len(fast.spike_times[0]) > 4 * len(slow.spike_times[0]) and ev.activity_stats(fast)[1] > 3 * ev.activity_stats(slow)[1]


def test_spread_shrinks_the_neuron_positions_towards_the_middle_and_the_angles_between_columns():
    assert sc.neuron_position(0, 1.0) == pytest.approx(C.NEURON_POSITIONS[0]) and sc.neuron_position(0, 0.0)[0] == 0.5
    wide, tight = ev.make_dataset(n=3, spread=1.0), ev.make_dataset(n=3, spread=0.1)
    assert ev.min_pair_angle(tight.A) < 0.5 * ev.min_pair_angle(wide.A)


def test_active_mask_matches_the_waveform_support():
    ds = ev.make_dataset(noise=0.0)
    for i in range(ds.n_neurons):
        peaks = ds.spike_times[i]
        assert ds.active[i][peaks].all() and ds.active[i].mean() < 0.15


def test_dense_background_is_optional_and_streams_are_independent_of_other_settings():
    a = ev.make_dataset(g=0)
    b = ev.make_dataset(g=1, kind="rhythm", n=5, m=3)
    assert b.S.shape[0] == 4 and b.kinds[-1] == "rhythm" and np.allclose(a.S[0], b.S[0]) and np.allclose(a.S[1], b.S[1])
    assert abs(ev.make_dataset(noise=0.5, g=0).noise_sigma - ev.make_dataset(noise=0.5, g=1).noise_sigma) < 1e-12


def test_constants_are_consistent():
    assert len(C.NEURON_SIGMAS) == len(C.NEURON_RATES) == len(C.NEURON_AMPLITUDES) == len(C.NEURON_POSITIONS) == C.N_NEURONS_MAX
    assert C.N_BACKGROUND_MAX == 1 and set(C.RECONSTRUCTIONS) == set(C.RECONSTRUCTION_LABELS)
