"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe sobi_presets.py in sobi-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import sca_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_neurons_slider": SettingSpec("m", int, C.DEFAULT_N_NEURONS, C.N_NEURONS_MIN, C.N_NEURONS_MAX),
    "n_electrodes_slider": SettingSpec("n", int, C.DEFAULT_N_ELECTRODES, C.N_ELECTRODES_MIN, C.N_ELECTRODES_MAX),
    "rate_slider": SettingSpec("rate", float, C.DEFAULT_RATE_SCALE, C.RATE_SCALE_MIN, C.RATE_SCALE_MAX),
    "spread_slider": SettingSpec("spread", float, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "n_background_slider": SettingSpec("g", int, C.DEFAULT_N_BACKGROUND, C.N_BACKGROUND_MIN, C.N_BACKGROUND_MAX),
    "kind_select": SettingSpec("kind", _choice(C.BACKGROUND_KINDS), C.DEFAULT_BACKGROUND_KIND),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "n_samples_slider": SettingSpec("T", int, C.DEFAULT_N_SAMPLES, C.N_SAMPLES_MIN, C.N_SAMPLES_MAX),
    "reconstruction_select": SettingSpec("rec", _choice(C.RECONSTRUCTIONS), C.DEFAULT_RECONSTRUCTION),
    "contrast_select": SettingSpec("contrast", _choice(C.CONTRASTS), C.DEFAULT_CONTRAST),
    "init_start_select": SettingSpec("start", int, C.DEFAULT_INIT_START, min(C.INIT_STARTS), max(C.INIT_STARTS)),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"m": "n_neurons_slider", "n": "n_electrodes_slider", "rate_scale": "rate_slider", "spread": "spread_slider", "g": "n_background_slider", "kind": "kind_select", "noise": "noise_slider",
               "n_samples": "n_samples_slider", "reconstruction": "reconstruction_select", "contrast": "contrast_select", "init_start": "init_start_select", "seed": "seed_input"}
KEPT = {"kind_select": "_kind_kept"}        # bei ausgeblendetem Regler bleibt sein Wert hier erhalten


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; die Art des Hintergrunds (bei 0 Hintergrundquellen ausgeblendet, dann vom Widget-Zustand gelöscht) kehrt zum zuletzt gewählten Wert zurück."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["init_start_select"] = int(min(C.INIT_STARTS, key=lambda s: abs(s - st.session_state.get("init_start_select", C.DEFAULT_INIT_START))))
    st.session_state["rate_slider"] = round(st.session_state.get("rate_slider", C.DEFAULT_RATE_SCALE) * 4) / 4
    st.session_state["spread_slider"] = round(st.session_state.get("spread_slider", C.DEFAULT_SPREAD) * 10) / 10
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    st.session_state["_kind_kept"] = C.PRESETS[name]["kind"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
