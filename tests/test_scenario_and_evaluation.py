"""Auswertung und die Aussagen der App als Tests: jede Zahl in den Hilfetexten, Tabellen und Presets ist hier über die festen Sweep-Datensätze belegt (Toleranzen bewusst weit)."""

import numpy as np
import pytest

import sca_constants as C
import sca_evaluation as ev


def _analyses(seeds=C.SWEEP_SEEDS, settings=ev.Settings(), **kw):
    return [ev.analyse(ev.make_dataset(seed=s, **kw), settings) for s in seeds]


def _mean(analyses, method, key="neuron_corr"):
    return float(np.mean([getattr(a.methods[method], key) for a in analyses]))


# --- Kennzahlen: Handinstanzen ---------------------------------------------------------------------------------------------------------------


def test_axis_angles_hand_instances():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(ev.axis_angles(A, A), 0.0) and np.allclose(ev.axis_angles(A, -A[:, ::-1]), 0.0)                   # Vorzeichen und Reihenfolge egal
    B = np.array([[np.cos(np.radians(10)), 0.0], [np.sin(np.radians(10)), 1.0]])
    assert np.allclose(sorted(ev.axis_angles(A, B)), [0.0, 10.0])
    assert np.isnan(ev.axis_angles(np.eye(3), np.eye(3)[:, :2])).sum() == 1                                                  # weniger Schätzungen als Spalten


def test_min_pair_angle_and_activity_stats_hand_instances():
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    assert abs(ev.min_pair_angle(A) - 45.0) < 1e-9 and np.isnan(ev.min_pair_angle(A[:, :1]))
    ds = ev.make_dataset(noise=0.0)
    frac, overlap = ev.activity_stats(ds)
    n_active = ds.active.sum(axis=0)
    assert abs(frac - (n_active >= 1).mean()) < 1e-12 and abs(overlap - (n_active >= 2).sum() / (n_active >= 1).sum()) < 1e-12


def test_chance_angle_is_deterministic_and_far_above_the_sca_error():
    A = ev.make_dataset().A[:, :4]
    assert ev.chance_angle_error(A) == ev.chance_angle_error(A) and 13.0 < ev.chance_angle_error(A) < 17.0
    assert ev.chance_angle_error(ev.make_dataset(n=3).A[:, :4]) > 20.0


def test_analysis_has_the_expected_structure_and_the_references_are_only_computed_when_asked():
    a = ev.analyse_for((4, 2, 0, "gauss", 4.0, 0.5, 0.2, 20000, 7), ev.Settings())
    assert set(a.methods) == {"sca", "single", "ica", "sobi"} and set(a.models) == {"sca", "ica", "sobi"}
    assert all(np.isfinite(x) for x in (a.ref_clean, a.ref_low_rate, a.ref_wide)) and np.isnan(a.ref_no_background)
    b = ev.analyse(ev.make_dataset(), ev.Settings())
    assert all(np.isnan(x) for x in (b.ref_clean, b.ref_low_rate, b.ref_wide, b.ref_no_background))
    assert np.isnan(b.methods["ica"].angle_error) and np.isfinite(b.methods["sca"].angle_error)


def test_reconstruction_setting_switches_the_main_sca_method():
    a = ev.analyse(ev.make_dataset(spread=0.1), ev.Settings(reconstruction="single"))
    assert a.models["sca"].reconstruction == "single" and np.allclose(a.methods["sca"].estimates, a.methods["single"].estimates)


# --- Verdict-Codes ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("params,code", [
    ((4, 2, 0, "gauss", 1.0, 1.0, 0.05), "sca_wins"),
    ((5, 3, 0, "gauss", 1.0, 1.0, 0.05), "sca_wins"),
    ((4, 6, 0, "gauss", 1.0, 1.0, 0.05), "enough_electrodes"),
    ((4, 2, 0, "gauss", 4.0, 1.0, 0.05), "overlap_high"),
    ((4, 2, 0, "gauss", 1.0, 0.1, 0.05), "close_neurons"),
    ((4, 3, 1, "rhythm", 1.0, 1.0, 0.05), "dense_background"),
    ((4, 3, 0, "gauss", 1.0, 1.0, 1.0), "noise"),
    ((4, 1, 0, "gauss", 1.0, 1.0, 0.05), "one_electrode"),
])
def test_verdict_codes_hold_on_several_datasets(params, code):
    for seed in (7,) + C.SWEEP_SEEDS:
        a = ev.analyse_for(params + (20000, seed), ev.Settings())
        assert ev.verdict(a)[1] == code, (seed, ev.verdict(a)[1])


def test_sweep_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED and len(C.SWEEP_SEEDS) >= 5


def test_sweep_and_scene_tables_have_the_expected_shape_and_are_deterministic():
    rows = ev.sweep("n_electrodes", values=(2, 6))
    assert rows == ev.sweep("n_electrodes", values=(2, 6)) and [r["x"] for r in rows] == [2, 6] and rows[0]["sca"] > rows[0]["ica"] + 0.3 and rows[1]["ica"] > 0.95
    assert all(set(r) >= {"x", "sca", "single", "ica", "sobi", "angle", "overlap", "sca_std"} for r in rows)
    scenes = ev.scene_table()
    assert [r["scene"] for r in scenes] == [s for s, _ in ev.SCENES] and all(r["sca_min"] <= r["sca"] <= r["sca_max"] for r in scenes)


# --- Aussagen der App ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,sca,ica,sobi", [(2, 0.95, 0.41, 0.39), (3, 0.98, 0.68, 0.68)])
def test_electrode_help_text(n, sca, ica, sobi):
    """Beleg für die Hilfe zu den Elektroden (4 Neuronen): 2 Elektroden 0.95 / 0.41 / 0.39, 3 Elektroden 0.98 / 0.68 / 0.68 (SCA / ICA / SOBI)."""
    a = _analyses(n=n)
    assert abs(_mean(a, "sca") - sca) < 0.03 and abs(_mean(a, "ica") - ica) < 0.03 and abs(_mean(a, "sobi") - sobi) < 0.03


def test_with_enough_electrodes_ica_suffices_and_sca_is_on_par_or_better():
    """Beleg für die Hilfe zu den Elektroden und das Preset 'Genug Elektroden': ab 4 Elektroden ICA 0.98, SCA 0.995, SOBI 0.978."""
    for n in (4, 6):
        a = _analyses(n=n)
        assert _mean(a, "ica") > 0.97 and _mean(a, "sca") >= _mean(a, "ica") - 0.001 and _mean(a, "sca") > 0.99
    a = _analyses(n=6)
    assert abs(_mean(a, "ica") - 0.981) < 0.01 and abs(_mean(a, "sobi") - 0.978) < 0.01 and abs(_mean(a, "sca") - 0.995) < 0.01


def test_one_electrode_finds_nothing():
    a = _analyses(n=1)
    assert _mean(a, "sca") == 0.0 and _mean(a, "single") == 0.0


@pytest.mark.parametrize("rate,expected", [(0.5, 0.98), (1.0, 0.95), (2.0, 0.91), (4.0, 0.83)])
def test_rate_help_text(rate, expected):
    """Beleg für die Hilfe zur Feuerrate (2 Elektroden, 4 Neuronen): 0.98 / 0.95 / 0.91 / 0.83 bei Faktor 0.5 / 1 / 2 / 4."""
    assert abs(_mean(_analyses(n=2, rate_scale=rate), "sca") - expected) < 0.03


def test_overlap_fractions_in_the_help_text():
    """Beleg: bei Faktor 1 sind an etwa 6 % der aktiven Zeitpunkte zwei Neuronen gleichzeitig aktiv, bei Faktor 4 an etwa 24 %."""
    one = np.mean([a.overlap_fraction for a in _analyses(rate_scale=1.0)])
    four = np.mean([a.overlap_fraction for a in _analyses(rate_scale=4.0)])
    assert 0.05 < one < 0.09 and 0.22 < four < 0.3


def test_close_neurons_help_text():
    """Beleg für die Hilfe zum Neuronen-Abstand und zur Rekonstruktion (2 Elektroden, Abstand 0.1): Richtungsfehler unter 0.2 Grad, L1 0.79, Einzelquelle 0.92."""
    a = _analyses(n=2, spread=0.1)
    assert max(x.methods["sca"].angle_error for x in a) < 0.2 and abs(_mean(a, "sca") - 0.79) < 0.04 and abs(_mean(a, "single") - 0.92) < 0.04
    assert max(x.min_pair_angle for x in a) < 8.0 and min(x.min_pair_angle for x in _analyses(n=2, spread=1.0)) > 8.0


def test_dense_background_help_text():
    """Beleg für die Hilfe zum Hintergrund und das Preset (3 Elektroden): ohne Hintergrund 0.98, Gauß 0.96, Rhythmus 0.90; Richtungsfehler Rhythmus 3-5 Grad, Gauß unter 2.5."""
    plain = _analyses(n=3)
    gauss = _analyses(n=3, g=1, kind="gauss")
    rhythm = _analyses(n=3, g=1, kind="rhythm")
    assert abs(_mean(plain, "sca") - 0.985) < 0.02 and abs(_mean(gauss, "sca") - 0.96) < 0.03 and abs(_mean(rhythm, "sca") - 0.90) < 0.03
    assert all(2.5 < x.methods["sca"].angle_error < 6.5 for x in rhythm) and all(x.methods["sca"].angle_error < 2.5 for x in gauss)
    assert abs(_mean(rhythm, "ica") - 0.60) < 0.04 and abs(_mean(rhythm, "sobi") - 0.37) < 0.05


@pytest.mark.parametrize("noise,sca,ica", [(0.0, 0.99, 0.68), (0.4, 0.93, 0.56), (1.0, 0.74, 0.37)])
def test_noise_help_text(noise, sca, ica):
    """Beleg für die Hilfe zum Rauschen (3 Elektroden, 4 Neuronen): SCA 0.99 / 0.93 / 0.74, ICA 0.68 / 0.56 / 0.37."""
    a = _analyses(n=3, noise=noise)
    assert abs(_mean(a, "sca") - sca) < 0.04 and abs(_mean(a, "ica") - ica) < 0.04


def test_strong_noise_preset_numbers():
    """Beleg für die Preset-Hilfe: Rauschen 1.0, 3 Elektroden: Richtungen auf 2-3 Grad, SOBI 0.35."""
    a = _analyses(n=3, noise=1.0)
    assert all(1.2 < x.methods["sca"].angle_error < 3.5 for x in a) and abs(_mean(a, "sobi") - 0.35) < 0.04


def test_length_help_text():
    """Beleg für die Hilfe zur Länge (2 Elektroden): 0.94 mit 5000, 0.95 mit 40000 Abtastwerten."""
    assert abs(_mean(_analyses(n=2, n_samples=5000), "sca") - 0.94) < 0.03 and abs(_mean(_analyses(n=2, n_samples=40000), "sca") - 0.95) < 0.03


def test_low_rate_reference_of_the_high_rate_preset():
    """Beleg für die Preset-Hilfe 'Hohe Feuerrate': bei einem Viertel der Feuerrate 0.97, bei Faktor 4 Richtungsfehler 2 Grad."""
    assert abs(_mean(_analyses(n=2, rate_scale=0.25), "sca") - 0.97) < 0.03
    assert all(1.2 < x.methods["sca"].angle_error < 2.8 for x in _analyses(n=2, rate_scale=4.0))


def test_default_scene_direction_error_is_far_below_chance():
    """Beleg für die Preset-Hilfe 'Vier Neuronen, zwei Elektroden' und die Verdict-Texte: Richtungen auf unter 1.5 Grad (Zufall 15), ohne Hintergrund überall unter 1.2 Grad bei 3+ Elektroden."""
    a = _analyses(n=2)
    assert all(x.methods["sca"].angle_error < 1.5 for x in a) and all(13.0 < x.chance_angle < 17.0 for x in a)
    for n in (3, 4, 6):
        assert all(x.methods["sca"].angle_error < 1.2 for x in _analyses(n=n))


def test_start_hardly_matters():
    """Beleg für die Hilfe zum Start: die Neuronen-Korrelation streut über die fünf Starts um höchstens 0.01."""
    ds = ev.make_dataset()
    values = [ev.analyse(ds, ev.Settings(init_start=s)).methods["sca"].neuron_corr for s in C.INIT_STARTS]
    assert max(values) - min(values) < 0.01


def test_five_neurons_with_three_electrodes_scene():
    a = _analyses(m=5, n=3)
    assert _mean(a, "sca") > 0.94 and _mean(a, "ica") < 0.6 and _mean(a, "sobi") < 0.6
