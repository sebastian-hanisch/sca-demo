"""Defaults, Slider-Grenzen und feste Szenario-Größen der SCA-Demo. Das Szenario ist das der ica-demo/sobi-demo (Neuronen, Wellenformen, Feuerraten, Mischung wortgleich);
neu sind der Feuerraten-Faktor (steuert die Überlappung), der Neuronen-Abstand (steuert die Ähnlichkeit der Mischrichtungen), der optionale dichte Hintergrund und die SCA-Regler."""

# --- Szenario (fest, wortgleich aus ica-demo/sobi-demo) ------------------------------------------------------------------------
SAMPLE_RATE = 10_000                       # Hz
WAVEFORM_LENGTH = 30                       # Abtastwerte je Spike
PEAK_INDEX = 8                             # Lage der negativen Spitze in der Wellenform
OVERSHOOT = 0.4                            # Höhe des positiven Nachschlags
REFRACTORY = 20                            # Abtastwerte (2 ms)
NEURON_SIGMAS = (2.0, 3.0, 2.5, 4.0, 3.5)          # Breite der Spitze je Neuron
NEURON_RATES = (20.0, 28.0, 35.0, 24.0, 31.0)      # Feuerrate in Hz (bei Feuerraten-Faktor 1)
NEURON_AMPLITUDES = (1.0, 0.8, 1.2, 0.7, 0.9)      # Spitzenamplitude an der nächsten Elektrode
NEURON_POSITIONS = ((0.10, 0.30), (0.35, 0.20), (0.60, 0.35), (0.85, 0.25), (0.50, 0.55))   # Elektroden liegen bei y = 0, x in [0, 1]
DISTANCE_EPS = 0.05
BACKGROUND_AMPLITUDE = 0.8
GAUSS_AR_BASE = 0.95                       # AR(1)-Koeffizient des Gauß-Hintergrunds
RHYTHM_BASE_HZ = 10.0                      # Frequenz des Rhythmus-Hintergrunds
BACKGROUND_KINDS = ("gauss", "rhythm")
BACKGROUND_LABELS = {"gauss": "Gauß-Rauschen (farbig, AR(1))", "rhythm": "Rhythmus (sinusförmig)"}
ACTIVE_THRESHOLD = 0.1                     # ein Neuron gilt an einem Zeitpunkt als aktiv, wenn seine Wellenform dort mehr als 10 % der Spitzenhöhe erreicht

# --- Regler ---------------------------------------------------------------------------------------------------------------------
DEFAULT_N_NEURONS = 4
N_NEURONS_MIN, N_NEURONS_MAX = 2, 5
DEFAULT_N_ELECTRODES = 2
N_ELECTRODES_MIN, N_ELECTRODES_MAX = 1, 8
DEFAULT_RATE_SCALE = 1.0
RATE_SCALE_MIN, RATE_SCALE_MAX = 0.25, 4.0
DEFAULT_SPREAD = 1.0
SPREAD_MIN, SPREAD_MAX = 0.1, 1.0                                        # Faktor auf den Abstand der Neuronen zur Mitte der Zeile (0.1 = dicht beieinander, ähnliche Mischrichtungen)
DEFAULT_N_BACKGROUND = 0
N_BACKGROUND_MIN, N_BACKGROUND_MAX = 0, 1
DEFAULT_BACKGROUND_KIND = "gauss"
DEFAULT_NOISE = 0.05
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_N_SAMPLES = 20_000
N_SAMPLES_MIN, N_SAMPLES_MAX = 5_000, 40_000
RECONSTRUCTIONS = ("l1", "single")
RECONSTRUCTION_LABELS = {"l1": "L1-Minimierung (mehrere aktive Quellen)", "single": "Einzelquelle (nur die beste Spalte)"}
DEFAULT_RECONSTRUCTION = "l1"
N_RESTARTS = 10                            # Neustarts der Achsen-k-Means
KMEANS_MAX_ITER = 100
CONTRASTS = ("logcosh", "exp", "cube")
CONTRAST_LABELS = {"logcosh": "log cosh (robust)", "exp": "Gauß-Ableitung (sehr robust)", "cube": "Kurtosis (u³)"}
DEFAULT_CONTRAST = "logcosh"
METHODS = ("symmetric", "deflation")
DEFAULT_METHOD = "symmetric"
INIT_STARTS = (1, 2, 3, 4, 5)
DEFAULT_INIT_START = 1
DEFAULT_SEED = 7
MAX_ITER = 200                             # FastICA
TOL = 1e-6
JD_MAX_SWEEPS = 100                        # SOBI (Vergleich): Jacobi-Sweeps
JD_TOL = 1e-8
SOBI_LAGS = tuple(range(2, 21, 2))         # Verzögerungen des SOBI-Vergleichs (die "mittlere" Menge der sobi-demo)

# --- Auswertung -----------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)            # feste Datensätze der Sweeps, getrennt vom Demo-Seed


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(m=DEFAULT_N_NEURONS, n=DEFAULT_N_ELECTRODES, g=DEFAULT_N_BACKGROUND, kind=DEFAULT_BACKGROUND_KIND, rate_scale=DEFAULT_RATE_SCALE, spread=DEFAULT_SPREAD, noise=DEFAULT_NOISE,
                n_samples=DEFAULT_N_SAMPLES, reconstruction=DEFAULT_RECONSTRUCTION, contrast=DEFAULT_CONTRAST, init_start=DEFAULT_INIT_START, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Vier Neuronen, zwei Elektroden": _preset(),
    "Genug Elektroden": _preset(n=6),
    "Hohe Feuerrate": _preset(rate_scale=4.0),
    "Neuronen dicht beieinander": _preset(spread=0.1),
    "Dichter Rhythmus-Hintergrund": _preset(n=3, g=1, kind="rhythm"),
    "Starkes Rauschen": _preset(n=3, noise=1.0),
}
PRESET_HELP = {
    "Vier Neuronen, zwei Elektroden": "Mehr Neuronen als Elektroden: die Mischung ist nicht umkehrbar, ICA (Korrelation 0.41) und SOBI (0.39) scheitern. SCA nutzt, dass die Neuronen selten gleichzeitig feuern - jeder Elektrodenvektor "
                                      "zeigt fast immer auf die Mischspalte eines einzigen Neurons: Mischrichtungen auf unter 1.5 Grad genau (Zufall: 15), Neuronen-Korrelation 0.95.",
    "Genug Elektroden": "Sechs Elektroden für vier Neuronen: die ICA reicht (0.981), SOBI ebenso (0.978). SCA ist trotzdem gleichauf oder besser (0.995), weil es das Rauschen an ruhigen Zeitpunkten weglässt - "
                        "es wird hier aber nicht gebraucht.",
    "Hohe Feuerrate": "Vierfache Feuerrate: an gut jedem vierten aktiven Zeitpunkt sind zwei Neuronen gleichzeitig aktiv, die Annahme 'höchstens eine aktive Quelle' wird brüchig. Die Richtungen bleiben gut gefunden (Fehler 2 Grad), "
                      "aber die Rekonstruktion leidet: Korrelation 0.83 statt 0.97 bei einem Viertel der Feuerrate. ICA und SOBI bleiben bei 0.41 - es ist weiter die falsche Elektrodenzahl.",
    "Neuronen dicht beieinander": "Alle vier Neuronen liegen nahe der Mitte der Zeile: ihre Mischrichtungen unterscheiden sich um wenige Grad. Die Richtungen findet SCA trotzdem (Fehler unter 0.2 Grad), aber die L1-Rekonstruktion verteilt "
                                  "Energie auf fast parallele Spalten: 0.79. Die Einzelquellen-Zuordnung ist hier besser (0.92).",
    "Dichter Rhythmus-Hintergrund": "Ein Sinusrhythmus wirkt dauernd auf alle Elektroden - er ist nicht sparse. Die Richtungen der Neuronen werden ungenauer (Fehler 3-5 Grad), die Korrelation fällt von 0.98 auf 0.90; "
                                    "ICA (0.60) und SOBI (0.37) bleiben bei drei Elektroden für fünf Quellen weit zurück.",
    "Starkes Rauschen": "Rauschen von 100 % des Neuronen-Signals bei drei Elektroden für vier Neuronen: die Richtungen sind noch auf 2-3 Grad genau, die Rekonstruktion leidet (0.74 statt 0.99 ohne Rauschen). ICA 0.38, SOBI 0.35.",
}
# Bänder (Seed des Presets; Werte mit dem ausgelieferten Code kalibriert, bewusst weit): Neuronen-Korrelation je Verfahren, Winkelfehler der SCA (Grad), Urteil
PRESET_EXPECTED_BANDS = {
    "Vier Neuronen, zwei Elektroden": {"sca": (0.88, 1.0), "ica": (0.3, 0.5), "sobi": (0.3, 0.5), "angle": (0.0, 3.0), "verdict": "sca_wins"},
    "Genug Elektroden": {"sca": (0.97, 1.0), "ica": (0.95, 1.0), "sobi": (0.95, 1.0), "verdict": "enough_electrodes"},
    "Hohe Feuerrate": {"sca": (0.7, 0.92), "ica": (0.3, 0.5), "angle": (0.0, 4.0), "verdict": "overlap_high"},
    "Neuronen dicht beieinander": {"sca": (0.65, 0.9), "single": (0.8, 1.0), "angle": (0.0, 1.0), "verdict": "close_neurons"},
    "Dichter Rhythmus-Hintergrund": {"sca": (0.8, 0.96), "ica": (0.5, 0.7), "sobi": (0.25, 0.5), "angle": (2.0, 8.0), "verdict": "dense_background"},
    "Starkes Rauschen": {"sca": (0.6, 0.86), "ica": (0.25, 0.5), "sobi": (0.2, 0.5), "verdict": "noise"},
}
