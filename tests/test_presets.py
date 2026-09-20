"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import sca_constants as C
from sca_evaluation import Settings, analyse_for, verdict


def _measure(p):
    params = (p["m"], p["n"], p["g"], p["kind"], p["rate_scale"], p["spread"], p["noise"], p["n_samples"], p["seed"])
    a = analyse_for(params, Settings(reconstruction=p["reconstruction"], contrast=p["contrast"], init_start=p["init_start"]))
    out = {"verdict": verdict(a)[1], "angle": a.methods["sca"].angle_error}
    for name in ("sca", "single", "ica", "sobi"):
        out[name] = a.methods[name].neuron_corr
    return out


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_NEURONS_MIN <= p["m"] <= C.N_NEURONS_MAX and C.N_ELECTRODES_MIN <= p["n"] <= C.N_ELECTRODES_MAX and C.N_BACKGROUND_MIN <= p["g"] <= C.N_BACKGROUND_MAX
        assert p["kind"] in C.BACKGROUND_KINDS and C.RATE_SCALE_MIN <= p["rate_scale"] <= C.RATE_SCALE_MAX and abs(p["rate_scale"] * 4 - round(p["rate_scale"] * 4)) < 1e-9
        assert C.SPREAD_MIN <= p["spread"] <= C.SPREAD_MAX and abs(p["spread"] * 10 - round(p["spread"] * 10)) < 1e-9 and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX
        assert C.N_SAMPLES_MIN <= p["n_samples"] <= C.N_SAMPLES_MAX and p["n_samples"] % 1000 == 0 and p["reconstruction"] in C.RECONSTRUCTIONS and p["contrast"] in C.CONTRASTS and p["init_start"] in C.INIT_STARTS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
