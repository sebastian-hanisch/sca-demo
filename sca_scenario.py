"""Mehrelektroden-Szenario der SCA-Demo: dasselbe Array wie in ica-demo und sobi-demo (Neuronen, Wellenformen, Feuern, Mischung, Rauschen - dort wortgleich übernommen), ohne Laufzeitverzögerung.
Neu: der Feuerraten-Faktor `rate_scale` (mehr Überlappung), die Aktivität je Neuron und Zeitpunkt (`active`) und ein optionaler **dichter** Hintergrund als Grenzfall.

Nur numpy. Jede Quelle hat einen eigenen Zufallsstrom (Seed, Nummer): Neuron 0 feuert für einen Seed immer gleich, egal wie viele Neuronen oder Elektroden eingestellt sind."""

from dataclasses import dataclass

import numpy as np

import sca_constants as C


@dataclass(frozen=True)
class Dataset:
    S: np.ndarray                 # (k, T) wahre Quellen, Zeilen 0..m-1 = Neuronen, danach Hintergrund; jede Zeile hat Varianz 1
    X: np.ndarray                 # (n, T) Elektrodensignale (mit Rauschen)
    X_clean: np.ndarray           # (n, T) ohne Rauschen
    A: np.ndarray                 # (n, k) Mischmatrix
    spike_times: tuple            # je Neuron: Zeitpunkte der negativen Spitze (Abtastwerte)
    active: np.ndarray            # (m, T) bool: Neuron ist an diesem Zeitpunkt aktiv (Wellenform über 10 % der Spitzenhöhe)
    kinds: tuple                  # je Quelle: "neuron", "gauss" oder "rhythm"
    noise_sigma: float
    n_neurons: int
    n_electrodes: int
    seed: int


def spike_waveform(index):
    """Biphasische Wellenform: tiefe negative Spitze bei PEAK_INDEX, flacher positiver Nachschlag."""
    sigma = C.NEURON_SIGMAS[index]
    t = np.arange(C.WAVEFORM_LENGTH, dtype=float)
    return -np.exp(-((t - C.PEAK_INDEX) / sigma) ** 2) + C.OVERSHOOT * np.exp(-((t - (C.PEAK_INDEX + 2.5 * sigma)) / (1.6 * sigma)) ** 2)


def firing_times(index, n_samples, seed, rate_scale=1.0):
    """Poisson-artiges Feuern mit Refraktärzeit: Abstände = Refraktärzeit + Exponentialverteilung. Spike-Startzeitpunkte (Abtastwerte)."""
    rng = np.random.default_rng([seed, index])
    mean_isi = C.SAMPLE_RATE / (C.NEURON_RATES[index] * rate_scale)
    n_draw = int(n_samples / (mean_isi - C.REFRACTORY) * 1.5) + 20
    isi = C.REFRACTORY + rng.exponential(mean_isi - C.REFRACTORY, n_draw)
    times = np.cumsum(isi) - isi[0] + rng.uniform(0, mean_isi)
    return times[times < n_samples - C.WAVEFORM_LENGTH].astype(int)


def neuron_source(index, n_samples, seed, rate_scale=1.0):
    starts = firing_times(index, n_samples, seed, rate_scale)
    impulses = np.zeros(n_samples)
    impulses[starts] = 1.0
    s = np.convolve(impulses, spike_waveform(index))[:n_samples]
    return s, starts + C.PEAK_INDEX


def background_source(kind, n_samples, seed):
    rng = np.random.default_rng([seed, 2000])
    if kind == "rhythm":
        t = np.arange(n_samples) / C.SAMPLE_RATE
        return np.sin(2 * np.pi * C.RHYTHM_BASE_HZ * t + rng.uniform(0, 2 * np.pi))
    e = rng.standard_normal(n_samples)
    s = np.empty(n_samples)
    s[0] = e[0]
    for t_ in range(1, n_samples):                                    # AR(1); n_samples <= 40000, einmalig je Datensatz
        s[t_] = C.GAUSS_AR_BASE * s[t_ - 1] + e[t_]
    return s


def electrode_positions(n):
    return np.array([[0.5, 0.0]]) if n == 1 else np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])


def neuron_position(i, spread=1.0):
    """Ort des Neurons i; `spread` verkleinert seinen Abstand zur Mitte der Zeile (x = 0.5), 1 = wie in ica-demo/sobi-demo."""
    x, y = C.NEURON_POSITIONS[i]
    return (0.5 + spread * (x - 0.5), y)


def mixing_matrix(m, n, g, spread=1.0):
    """(n, m+g): Neuronen mit 1/(d^2+eps)-Abfall (Spitzenamplitude je Neuron), Hintergrund als flächiges Feld (gleich stark)."""
    pos = electrode_positions(n)
    cols = []
    for i in range(m):
        d = np.linalg.norm(pos - np.array(neuron_position(i, spread)), axis=1)
        a = 1.0 / (d ** 2 + C.DISTANCE_EPS)
        cols.append(C.NEURON_AMPLITUDES[i] * a / a.max())
    for _ in range(g):
        cols.append(C.BACKGROUND_AMPLITUDE * np.ones(n))
    return np.column_stack(cols)


def make_dataset(n_neurons, n_electrodes, n_background, background_kind, rate_scale, spread, noise, n_samples, seed):
    m, n, g = n_neurons, n_electrodes, n_background
    sources, spikes, kinds, raw = [], [], [], []
    for i in range(m):
        s, peaks = neuron_source(i, n_samples, seed, rate_scale)
        raw.append(s)
        sources.append(s), spikes.append(peaks), kinds.append("neuron")
    for _ in range(g):
        sources.append(background_source(background_kind, n_samples, seed)), kinds.append(background_kind)
    S = np.array(sources)
    S = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    active = np.abs(np.array(raw)) > C.ACTIVE_THRESHOLD
    A = mixing_matrix(m, n, g, spread)
    X_clean = A @ S
    sigma = noise * (A[:, :m] @ S[:m]).std()                            # Rauschen relativ zum Neuronen-Signal: unabhängig vom Hintergrund
    X = X_clean + sigma * np.random.default_rng([seed, 1000]).standard_normal(X_clean.shape)
    return Dataset(S, X, X_clean, A, tuple(spikes), active, tuple(kinds), float(sigma), m, n, seed)
