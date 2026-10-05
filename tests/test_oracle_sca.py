"""Orakel (anderer Rechenweg als der eigene Code): Brute Force über alle Zuordnungen (Achsen-k-Means auf kleinen 2-D-Instanzen), SciPy (Zuordnung, LP-Löser
für die L1-Rekonstruktion), Definitionsschleifen (aktive Zeitpunkte, Einzelquelle, Überlappung, Zufallsniveau per Permutationen), scikit-learn (FastICA-Kopie)
und eine Schleifen-Nachbildung des Szenarios. Nur kleine Instanzen, wenige Sekunden."""

import itertools
import math
import warnings

import numpy as np
import pytest

import sca_algorithm as alg
import sca_constants as C
import sca_evaluation as ev
import sca_ica as ica
import sca_scenario as sc

sp_opt = pytest.importorskip("scipy.optimize")
sk_decomp = pytest.importorskip("sklearn.decomposition")


def _objective(U, labels, k):
    """Zielwert für eine feste Zuordnung: Achse je Cluster = Hauptvektor seiner Streumatrix (eigh), Wert = Mittel von max_j |u·c_j|."""
    axes = []
    for j in range(k):
        members = U[labels == j]
        if len(members) == 0:
            return -1.0
        axes.append(np.linalg.eigh(members.T @ members)[1][:, -1])
    return float(np.abs(U @ np.array(axes).T).max(axis=1).mean())


@pytest.mark.parametrize("k,seed", [(2, 0), (2, 1), (3, 2)])
def test_axial_kmeans_reaches_the_brute_force_optimum_on_tiny_instances(k, seed):
    rng = np.random.default_rng(seed)
    N = 9
    ang = rng.uniform(0, math.pi, size=k)
    th = ang[rng.integers(0, k, N)] + rng.normal(0, 0.05, N)
    U = np.column_stack([np.cos(th), np.sin(th)]) * rng.choice([-1, 1], N)[:, None]
    best = max(_objective(U, np.array(lab), k) for lab in itertools.product(range(k), repeat=N))
    r = alg.axial_kmeans(U, k, n_restarts=10, seed=seed)
    assert r.objective == pytest.approx(best, abs=1e-4)
    assert r.objective == pytest.approx(float(np.abs(U @ r.centers.T).max(axis=1).mean()), abs=1e-12)


def test_axial_kmeans_fixed_point_labels_are_nearest_axes_and_centres_are_principal_axes():
    rng = np.random.default_rng(5)
    ang = np.radians([10.0, 55.0, 100.0])
    th = ang[rng.integers(0, 3, 300)] + rng.normal(0, 0.03, 300)
    U = np.column_stack([np.cos(th), np.sin(th)]) * rng.choice([-1, 1], 300)[:, None]
    r = alg.axial_kmeans(U, 3, n_restarts=4, seed=1)
    assert r.converged
    assert np.array_equal(np.abs(U @ r.centers.T).argmax(axis=1), r.labels)
    for j in range(3):
        v = np.linalg.eigh(U[r.labels == j].T @ U[r.labels == j])[1][:, -1]
        assert abs(abs(v @ r.centers[j]) - 1.0) < 1e-8


def test_active_and_core_samples_equal_the_definition_by_loops():
    ds = ev.make_dataset(seed=3)
    D = ds.X - np.median(ds.X, axis=1, keepdims=True)
    r = np.array([math.sqrt(sum(D[j, t] ** 2 for j in range(D.shape[0]))) for t in range(D.shape[1])])
    thr = max(4 * np.quantile(r, 0.2), 0.1 * np.quantile(r, 0.995))
    assert np.array_equal(r > thr, alg.active_samples(ds.X))
    assert np.array_equal((r > thr) & (r > 0.3 * np.quantile(r, 0.995)), alg.core_samples(ds.X))


def test_single_reconstruction_equals_projection_on_the_best_matching_column():
    rng = np.random.default_rng(6)
    A = np.abs(rng.standard_normal((3, 5))) + 0.05
    V = rng.standard_normal((3, 40))
    R = alg.single_reconstruct(A, V)
    for t in range(40):
        cos = [abs(A[:, j] @ V[:, t]) / np.linalg.norm(A[:, j]) / np.linalg.norm(V[:, t]) for j in range(5)]
        w = int(np.argmax(cos))
        ref = np.zeros(5)
        ref[w] = A[:, w] @ V[:, t] / np.linalg.norm(A[:, w])
        assert np.allclose(R[:, t], ref, atol=1e-10)


def test_l1_reconstruction_at_demo_settings_is_close_to_the_exact_linear_program():
    """Standardeinstellungen der Demo (30 Iterationen) gegen die exakte L1-Lösung (scipy linprog): L1-Norm höchstens 3 % darüber, Nebenbedingung erfüllt."""
    rng = np.random.default_rng(7)
    A = np.abs(rng.standard_normal((3, 5))) + 0.1
    A /= np.linalg.norm(A, axis=0)
    S = np.zeros((5, 25))
    for t in range(25):
        S[rng.integers(0, 5), t] = rng.standard_normal() * 2 + 1.5
    X = A @ S
    ours = alg.l1_reconstruct(A, X, ridge=1e-9)
    for t in range(25):
        res = sp_opt.linprog(np.ones(10), A_eq=np.hstack([A, -A]), b_eq=X[:, t], bounds=(0, None), method="highs")
        assert np.abs(ours[:, t]).sum() <= 1.03 * res.fun + 1e-9
        assert np.abs(A @ ours[:, t] - X[:, t]).max() < 1e-4


def test_axis_angles_equal_scipy_assignment_and_min_pair_angle_equals_loops():
    rng = np.random.default_rng(8)
    for _ in range(25):
        n, k, kh = int(rng.integers(2, 6)), int(rng.integers(2, 6)), int(rng.integers(1, 6))
        A, B = np.abs(rng.standard_normal((n, k))) + 0.05, np.abs(rng.standard_normal((n, kh))) + 0.05
        cm = np.abs((A / np.linalg.norm(A, axis=0)).T @ (B / np.linalg.norm(B, axis=0)))
        r, c = sp_opt.linear_sum_assignment(cm, maximize=True)
        ref = np.full(k, np.nan)
        for i, j in zip(r, c):
            ref[i] = math.degrees(math.acos(min(1.0, cm[i, j])))
        assert np.allclose(ev.axis_angles(A, B), ref, equal_nan=True, atol=1e-7)
        mp = min(math.degrees(math.acos(min(1.0, abs(A[:, i] @ A[:, j]) / np.linalg.norm(A[:, i]) / np.linalg.norm(A[:, j])))) for i in range(k) for j in range(k) if i != j)
        assert ev.min_pair_angle(A) == pytest.approx(mp, abs=1e-6)


def test_chance_level_equals_a_permutation_brute_force_simulation():
    ds = ev.make_dataset(n=2, seed=100000)
    A = ds.A[:, :4]
    An = A / np.linalg.norm(A, axis=0)
    r = np.random.default_rng(1)
    errs = []
    for _ in range(400):
        B = np.abs(r.standard_normal((2, 4)))
        B /= np.linalg.norm(B, axis=0)
        cm = np.abs(An.T @ B)
        bp = max(itertools.permutations(range(4)), key=lambda p: sum(cm[i, p[i]] for i in range(4)))
        errs.append(np.mean([math.degrees(math.acos(min(1.0, cm[i, bp[i]]))) for i in range(4)]))
    assert ev.chance_angle_error(A) == pytest.approx(np.mean(errs), abs=1.0)


def test_activity_stats_equal_loops():
    ds = ev.make_dataset(seed=7)
    na = ds.active.sum(axis=0)
    any1 = sum(1 for t in range(len(na)) if na[t] >= 1)
    any2 = sum(1 for t in range(len(na)) if na[t] >= 2)
    frac, over = ev.activity_stats(ds)
    assert frac == pytest.approx(any1 / len(na), abs=1e-12) and over == pytest.approx(any2 / max(any1, 1), abs=1e-12)


def test_fastica_copy_equals_scikit_learn_with_the_same_start():
    ds = ev.make_dataset(n=6)
    ours = ica.fit_ica(ds.X, 4, "logcosh", "symmetric", init_start=3, max_iter=500, tol=1e-9)
    W0 = np.random.default_rng([3, 4242]).standard_normal((4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sk_decomp.FastICA(algorithm="parallel", whiten=False, fun="logcosh", w_init=W0, tol=1e-9, max_iter=500).fit(ours.whitening.Z.T)
    assert np.allclose(ours.W, ref.components_, atol=1e-8)


@pytest.mark.parametrize("m,n,g,kind,rate,spread", [(4, 2, 0, "gauss", 1.0, 1.0), (5, 3, 1, "rhythm", 4.0, 0.1), (3, 6, 1, "gauss", 0.5, 0.5)])
def test_dataset_equals_a_loop_rebuild(m, n, g, kind, rate, spread):
    T, seed = 6000, 13
    ds = sc.make_dataset(m, n, g, kind, rate, spread, 0.0, T, seed)
    pos = [0.5] if n == 1 else list(np.linspace(0, 1, n))
    A = np.zeros((n, m + g))
    for i in range(m):
        x0, y0 = C.NEURON_POSITIONS[i]
        px = 0.5 + spread * (x0 - 0.5)
        d = np.array([math.hypot(p - px, y0) for p in pos])
        a = 1.0 / (d ** 2 + C.DISTANCE_EPS)
        A[:, i] = C.NEURON_AMPLITUDES[i] * a / a.max()
    for j in range(g):
        A[:, m + j] = C.BACKGROUND_AMPLITUDE
    assert np.allclose(ds.A, A) and np.allclose(ds.X_clean, A @ ds.S)
    for i in range(m):
        starts = sc.firing_times(i, T, seed, rate)
        w = sc.spike_waveform(i)
        s = np.zeros(T)
        for st in starts:
            for k in range(C.WAVEFORM_LENGTH):
                if st + k < T:
                    s[st + k] += w[k]
        assert np.allclose((s - s.mean()) / s.std(), ds.S[i], atol=1e-9)
        assert np.array_equal(ds.active[i], np.abs(s) > C.ACTIVE_THRESHOLD)
        assert np.array_equal(ds.spike_times[i], starts + C.PEAK_INDEX)
