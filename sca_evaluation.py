"""Auswertung der SCA-Demo: Zuordnung, Kennzahlen (Korrelation, SIR, Amari-Index, Spike-Treffer) - aus ica-demo/sobi-demo übernommen - und die neuen Größen (Winkelfehler der Mischspalten
mit simuliertem Zufallsniveau, Überlappung, Ähnlichkeit der Richtungen), Sweeps, Szenen und das Urteil."""

from dataclasses import dataclass

import numpy as np

import sca_algorithm as alg
import sca_constants as C
import sca_ica as ica
import sca_scenario as sc
import sca_sobi as sobi


@dataclass(frozen=True)
class Settings:
    reconstruction: str = C.DEFAULT_RECONSTRUCTION
    contrast: str = C.DEFAULT_CONTRAST
    init_start: int = C.DEFAULT_INIT_START


def make_dataset(m=C.DEFAULT_N_NEURONS, n=C.DEFAULT_N_ELECTRODES, g=C.DEFAULT_N_BACKGROUND, kind=C.DEFAULT_BACKGROUND_KIND, rate_scale=C.DEFAULT_RATE_SCALE, spread=C.DEFAULT_SPREAD, noise=C.DEFAULT_NOISE,
                 n_samples=C.DEFAULT_N_SAMPLES, seed=C.DEFAULT_SEED):
    return sc.make_dataset(m, n, g, kind, rate_scale, spread, noise, n_samples, seed)


def n_components(ds):
    """Die Zahl der Quellen wird als bekannt angenommen (Standardannahme von FastICA und SOBI); bei weniger Elektroden als Quellen bleibt nur n."""
    return min(ds.n_electrodes, ds.S.shape[0])


def correlation_matrix(S, estimates):
    """|Korrelation| (k, nc) zwischen wahren Quellen und Schätzungen."""
    a = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    b = estimates - estimates.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    b = b / np.where(sd > 0, sd, 1.0)
    return np.abs(a @ b.T) / S.shape[1]


def assign(C_abs):
    """Optimale Zuordnung Quelle -> Schätzung (jede Schätzung höchstens einer Quelle), Summe der |Korrelationen| maximal. Exakt per Bitmasken-DP.
    Rückgabe: Liste je Quelle mit dem Index der Schätzung oder -1 (nicht zugeordnet, nur wenn es weniger Schätzungen als Quellen gibt)."""
    k, nc = C_abs.shape
    best = {}

    def solve(row, used):
        if row == k:
            return 0.0, ()
        key = (row, used)
        if key in best:
            return best[key]
        result = (solve(row + 1, used)[0], (-1,) + solve(row + 1, used)[1])
        for j in range(nc):
            if not used >> j & 1:
                value, rest = solve(row + 1, used | 1 << j)
                if value + C_abs[row, j] > result[0] + 1e-12:
                    result = (value + C_abs[row, j], (j,) + rest)
        best[key] = result
        return result

    return list(solve(0, 0)[1])


def matched(S, estimates):
    """Zuordnung + Vorzeichen-/Skalenkorrektur per Regression: (Zuordnung, |Korrelation| je Quelle, Schätzquellen in Skala und Vorzeichen der wahren Quellen)."""
    cm = correlation_matrix(S, estimates)
    idx = assign(cm)
    corr = np.array([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    aligned = np.zeros_like(S)
    for i, j in enumerate(idx):
        if j >= 0:
            e = estimates[j] - estimates[j].mean()
            aligned[i] = (e @ (S[i] - S[i].mean()) / (e @ e)) * e + S[i].mean()
    return idx, corr, aligned


def sir_db(corr):
    """Signal-zu-Interferenz in dB aus der Korrelation: rho^2 / (1 - rho^2)."""
    r2 = np.clip(np.asarray(corr) ** 2, 1e-9, 1.0 - 1e-9)
    return 10.0 * np.log10(r2 / (1.0 - r2))


def amari_index(P):
    """Amari-Index einer (nc, k)-Matrix P = Entmischung x Mischung; 0 = perfekt (nur Permutation und Skalierung), 1 = schlechtestmöglich; verallgemeinert auf nicht quadratische P."""
    P = np.abs(P)
    k = P.shape[1]
    rows = (P.sum(axis=1) / P.max(axis=1) - 1.0).sum()
    cols = (P.sum(axis=0) / P.max(axis=0) - 1.0).sum()
    d = max(k, P.shape[0])
    return float((rows + cols) / (2.0 * d * (d - 1))) if d > 1 else 0.0


# --- Spike-Erkennung auf den Schätzquellen ------------------------------------------------------------------------------------
DETECT_MIN_SEPARATION = 15         # Abtastwerte zwischen zwei erkannten Spitzen
DETECT_TOLERANCE = 4               # erlaubte Abweichung zur wahren Spitze
DETECT_SIGMA_FACTOR = 4.0          # Schwelle: 4 robuste Standardabweichungen (MAD) ...
DETECT_DEPTH_FRACTION = 0.3        # ... mindestens aber 30 % der typischen Spitzentiefe (sonst würde ein rauschfreies Signal jeden Rest melden)


def detect_spikes(x):
    """Negative Spitzen von x (Vorzeichen bereits wie beim Neuron): tiefste zuerst, Mindestabstand; Schwelle max(4 sigma_MAD, 0.3 x typische Tiefe)."""
    sigma = np.median(np.abs(x - np.median(x))) / 0.6745
    threshold = -DETECT_SIGMA_FACTOR * sigma
    taken = np.zeros(len(x), bool)
    peaks = []
    for t in np.argsort(x):
        if x[t] > threshold:
            break
        if taken[max(0, t - DETECT_MIN_SEPARATION): t + DETECT_MIN_SEPARATION + 1].any():
            continue
        taken[t] = True
        peaks.append(t)
        if len(peaks) == 10:                                             # typische Tiefe = Median der 10 tiefsten Spitzen
            threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
    return np.sort(np.array(peaks, dtype=int))


def spike_f1(detected, true):
    """F1 der erkannten gegen die wahren Spitzenzeiten (Zuordnung je wahrer Spitze zur nächsten, jede erkannte höchstens einmal)."""
    if len(detected) == 0 or len(true) == 0:
        return 0.0
    used = np.zeros(len(detected), bool)
    tp = 0
    for t in true:
        d = np.abs(detected - t)
        j = int(np.argmin(d))
        if d[j] <= DETECT_TOLERANCE and not used[j]:
            used[j] = True
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(detected), tp / len(true)
    return 2 * precision * recall / (precision + recall)



def excess_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
    return (x ** 4).mean(axis=-1) - 3.0






# --- Mischspalten: Winkelfehler, Zufallsniveau, Überlappung -------------------------------------------------------------------------------


def axis_angles(A, A_hat):
    """Winkel (Grad) zwischen wahren und geschätzten Mischspalten (Achsen, Vorzeichen egal) nach optimaler Zuordnung per |cos|, je wahrer Spalte; NaN, wenn keine Schätzung zugeordnet ist."""
    an = A / np.linalg.norm(A, axis=0, keepdims=True)
    bn = A_hat / np.linalg.norm(A_hat, axis=0, keepdims=True)
    cm = np.abs(an.T @ bn)
    idx = assign(cm)
    return np.array([np.degrees(np.arccos(np.clip(cm[i, j], -1.0, 1.0))) if j >= 0 else float("nan") for i, j in enumerate(idx)])


def chance_angle_error(A, n_draws=300, seed=0):
    """Zufallsniveau des mittleren Winkelfehlers: zufällig gleichverteilte Achsen (im positiven Orthanten, wie die Mischspalten), optimal zugeordnet; Mittel über `n_draws` Ziehungen."""
    rng = np.random.default_rng([seed, 31])
    n, k = A.shape
    errors = []
    for _ in range(n_draws):
        B = np.abs(rng.standard_normal((n, k)))
        errors.append(np.nanmean(axis_angles(A, B)))
    return float(np.mean(errors))


def min_pair_angle(A):
    """Kleinster Winkel (Grad) zwischen zwei verschiedenen Mischspalten: je kleiner, desto ähnlicher die Richtungen."""
    an = A / np.linalg.norm(A, axis=0, keepdims=True)
    c = np.abs(an.T @ an)
    np.fill_diagonal(c, 0.0)
    return float(np.degrees(np.arccos(np.clip(c.max(), -1.0, 1.0)))) if A.shape[1] > 1 else float("nan")


def activity_stats(ds):
    """(aktiver Anteil der Zeitpunkte, Überlappungsanteil): Anteil der Zeitpunkte mit mindestens einem aktiven Neuron; Anteil davon mit mindestens zwei."""
    n_active = ds.active.sum(axis=0)
    any_active = (n_active >= 1).sum()
    return float((n_active >= 1).mean()), float((n_active >= 2).sum() / max(any_active, 1))


# --- Kennzahlen je Verfahren -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    name: str
    idx: list                     # je Quelle: Index der zugeordneten Schätzung (-1 = keine)
    corr: np.ndarray              # |Korrelation| je Quelle
    aligned: np.ndarray           # (k, T) zugeordnete Schätzungen in Vorzeichen und Skala der wahren Quellen
    neuron_corr: float
    neuron_sir: float
    f1: float
    angle_error: float            # mittlerer Winkelfehler der Neuronen-Mischspalten in Grad (nur SCA; sonst NaN)
    estimates: np.ndarray         # (nc, T) rohe Schätzungen


def evaluate_method(name, ds, estimates, A_hat=None):
    idx, corr, aligned = matched(ds.S, estimates)
    m = ds.n_neurons
    f1 = float(np.mean([spike_f1(detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(m)]))
    angle = float(np.nanmean(axis_angles(ds.A, A_hat)[:m])) if A_hat is not None else float("nan")
    return Method(name, idx, corr, aligned, float(corr[:m].mean()), float(sir_db(corr[:m]).mean()), f1, angle, estimates)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    models: dict                  # "sca", "ica", "sobi"
    methods: dict                 # "sca", "single", "ica", "sobi" -> Method
    active_fraction: float
    overlap_fraction: float
    min_pair_angle: float
    chance_angle: float
    ref_clean: float              # SCA-Neuronenkorrelation ohne Rauschen (nur bei Rauschen > 0)
    ref_low_rate: float           # ... bei Feuerraten-Faktor 0.25 (nur bei Faktor > 1)
    ref_wide: float               # ... bei Neuronen-Abstand 1 (nur bei Abstand < 1)
    ref_no_background: float      # ... ohne Hintergrund (nur mit Hintergrund)


def _sca_neuron_corr(ds, settings):
    k = ds.S.shape[0]
    model = alg.fit_sca(ds.X, k, settings.reconstruction, seed=settings.init_start)
    return float(matched(ds.S, model.sources)[1][:ds.n_neurons].mean())


def _spec(ds):
    g = ds.S.shape[0] - ds.n_neurons
    return dict(m=ds.n_neurons, n=ds.n_electrodes, g=g, kind=ds.kinds[-1] if g else C.DEFAULT_BACKGROUND_KIND, n_samples=ds.S.shape[1], seed=ds.seed)


def analyse(ds, settings, rate_scale=None, spread=None, noise=None):
    """`rate_scale`, `spread`, `noise` (die Einstellungen, mit denen `ds` erzeugt wurde) werden nur für die Referenzläufe gebraucht; ohne Angabe entfallen die Referenzen."""
    k = ds.S.shape[0]
    nc = n_components(ds)
    sca_model = alg.fit_sca(ds.X, k, "l1", seed=settings.init_start)
    if settings.reconstruction != "l1":
        sca_model = alg.with_reconstruction(sca_model, ds.X, settings.reconstruction)
    single_model = alg.with_reconstruction(sca_model, ds.X, "single")
    ica_model = ica.fit_ica(ds.X, nc, settings.contrast, C.DEFAULT_METHOD, settings.init_start)
    sobi_model = sobi.fit_sobi(ds.X, nc, C.SOBI_LAGS)
    methods = {
        "sca": evaluate_method("sca", ds, sca_model.sources, sca_model.A_hat),
        "single": evaluate_method("single", ds, single_model.sources, single_model.A_hat),
        "ica": evaluate_method("ica", ds, ica_model.sources),
        "sobi": evaluate_method("sobi", ds, sobi_model.sources),
    }
    active_fraction, overlap_fraction = activity_stats(ds)
    nan = float("nan")
    ref_clean = ref_low_rate = ref_wide = ref_no_background = nan
    spec = _spec(ds)
    defaults = dict(rate_scale=C.DEFAULT_RATE_SCALE, spread=C.DEFAULT_SPREAD, noise=C.DEFAULT_NOISE)
    have = dict(rate_scale=rate_scale, spread=spread, noise=noise)
    if all(v is not None for v in have.values()):
        if noise > 0:
            ref_clean = _sca_neuron_corr(make_dataset(**{**spec, **have, "noise": 0.0}), settings)
        if rate_scale > 1:
            ref_low_rate = _sca_neuron_corr(make_dataset(**{**spec, **have, "rate_scale": 0.25}), settings)
        if spread < 1:
            ref_wide = _sca_neuron_corr(make_dataset(**{**spec, **have, "spread": 1.0}), settings)
        if spec["g"] >= 1:
            ref_no_background = _sca_neuron_corr(make_dataset(**{**spec, **have, "g": 0}), settings)
    return Analysis(ds, settings, {"sca": sca_model, "ica": ica_model, "sobi": sobi_model}, methods, active_fraction, overlap_fraction, min_pair_angle(ds.A[:, :ds.n_neurons]),
                    chance_angle_error(ds.A[:, :ds.n_neurons]), ref_clean, ref_low_rate, ref_wide, ref_no_background)


def analyse_for(params, settings):
    """Analyse mit allen Referenzläufen; `params` = (m, n, g, kind, rate_scale, spread, noise, n_samples, seed)."""
    m, n, g, kind, rate_scale, spread, noise, n_samples, seed = params
    return analyse(make_dataset(m, n, g, kind, rate_scale, spread, noise, n_samples, seed), settings, rate_scale, spread, noise)


# --- Sweeps und Szenen ---------------------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n_electrodes": (1, 2, 3, 4, 5, 6, 8),
    "rate_scale": (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    "noise": (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
    "spread": (0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    "n_neurons": (2, 3, 4, 5),
}
SWEEP_LABELS = {"n_electrodes": "Anzahl Elektroden", "rate_scale": "Feuerrate (Faktor)", "noise": "Rauschen (relativ zum Neuronen-Signal)", "spread": "Neuronen-Abstand (1 = wie gewohnt, klein = dicht beieinander)",
                "n_neurons": "Anzahl Neuronen"}
_SWEEP_KEYWORD = {"n_electrodes": "n", "rate_scale": "rate_scale", "noise": "noise", "spread": "spread", "n_neurons": "m"}
SWEEP_METHODS = ("sca", "single", "ica", "sobi")


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel und Streuung (über die festen Sweep-Datensätze) der Neuronen-Korrelation aller Verfahren und des SCA-Winkelfehlers in Abhängigkeit von einem Regler."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        per_seed = []
        for seed in C.SWEEP_SEEDS:
            kw = dict(base)
            kw[_SWEEP_KEYWORD[parameter]] = x
            a = analyse(make_dataset(seed=seed, **kw), settings)
            per_seed.append({**{name: a.methods[name].neuron_corr for name in SWEEP_METHODS}, "angle": a.methods["sca"].angle_error, "overlap": a.overlap_fraction})
        row = {"x": x}
        for key in SWEEP_METHODS + ("angle", "overlap"):
            arr = np.array([r[key] for r in per_seed], dtype=float)
            row[key] = float(np.nanmean(arr)) if not np.isnan(arr).all() else float("nan")
            row[key + "_std"] = float(np.nanstd(arr)) if not np.isnan(arr).all() else float("nan")
        rows.append(row)
    return rows


SCENES = (
    ("Genug Elektroden (4 Neuronen, 6 Elektroden)", dict(m=4, n=6)),
    ("Weniger Elektroden als Neuronen (4 Neuronen, 2 Elektroden)", dict(m=4, n=2)),
    ("Fünf Neuronen, drei Elektroden", dict(m=5, n=3)),
    ("Hohe Feuerrate (Faktor 4, 4 Neuronen, 3 Elektroden)", dict(m=4, n=3, rate_scale=4.0)),
    ("Neuronen dicht beieinander (Abstand 0.1, 4 Neuronen, 3 Elektroden)", dict(m=4, n=3, spread=0.1)),
    ("Dichter Rhythmus-Hintergrund (4 Neuronen, 3 Elektroden)", dict(m=4, n=3, g=1, kind="rhythm")),
    ("Starkes Rauschen (Rauschen 1.0, 4 Neuronen, 3 Elektroden)", dict(m=4, n=3, noise=1.0)),
)


def scene_table(settings=Settings(), **base):
    """Wer trennt was: je Szene die Neuronen-Korrelation von SCA (L1 und Einzelquelle), ICA und SOBI, Mittel und Spanne über die Sweep-Datensätze. `base` gilt für alles, was die Szene nicht festlegt."""
    rows = []
    for label, scene in SCENES:
        kw = dict(base)
        kw.update(scene)
        acc = {name: [] for name in SWEEP_METHODS}
        angles = []
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=seed, **kw), settings)
            for name in SWEEP_METHODS:
                acc[name].append(a.methods[name].neuron_corr)
            angles.append(a.methods["sca"].angle_error)
        row = {"scene": label, "angle": float(np.nanmean(angles)) if not np.isnan(angles).all() else float("nan")}
        for name, vals in acc.items():
            row[name] = float(np.mean(vals))
            row[name + "_min"], row[name + "_max"] = float(np.min(vals)), float(np.max(vals))
        rows.append(row)
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

VERDICT_DROP = 0.08               # Verlust der SCA-Neuronenkorrelation gegenüber dem Referenzlauf, ab dem eine Ursache benannt wird
VERDICT_NOISE_DROP = 0.12         # ... beim Rauschen (Rauschen wirkt schleichender)
VERDICT_WIN_MARGIN = 0.15         # SCA-Vorsprung vor ICA und SOBI
VERDICT_DENSE_ANGLE = 2.5         # Winkelfehler (Grad), ab dem ein dauerhaft aktiver Hintergrund als Ursache gilt (gemessen: Gauß-Hintergrund 1.1-1.8, Rhythmus 3.3-5.0, ohne Hintergrund unter 1.2)


def spec_background(ds):
    return ds.S.shape[0] > ds.n_neurons


def verdict(a):
    """(Art, Code, Kennzahlen). Nur mit großer Marge - kein Urteil nahe an einer Schwelle. Bei mehreren Ursachen gewinnt die mit dem größten Verlust."""
    ds = a.ds
    k, m = ds.S.shape[0], ds.n_neurons
    sca_m, single_m, ica_m, sobi_m = (a.methods[x] for x in ("sca", "single", "ica", "sobi"))
    data = {"sca": sca_m.neuron_corr, "single": single_m.neuron_corr, "ica": ica_m.neuron_corr, "sobi": sobi_m.neuron_corr, "f1": sca_m.f1, "ica_f1": ica_m.f1, "angle": sca_m.angle_error,
            "chance_angle": a.chance_angle, "overlap": a.overlap_fraction, "active": a.active_fraction, "min_angle": a.min_pair_angle, "ref_clean": a.ref_clean, "ref_low_rate": a.ref_low_rate,
            "ref_wide": a.ref_wide, "ref_no_background": a.ref_no_background, "k": k, "n": ds.n_electrodes}
    if ds.n_electrodes < 2:
        return "warning", "one_electrode", data
    drops = {
        "dense_background": a.ref_no_background - sca_m.neuron_corr if not np.isnan(a.ref_no_background) else 0.0,
        "noise": (a.ref_clean - sca_m.neuron_corr if not np.isnan(a.ref_clean) else 0.0) - (VERDICT_NOISE_DROP - VERDICT_DROP),
        "overlap_high": a.ref_low_rate - sca_m.neuron_corr if not np.isnan(a.ref_low_rate) else 0.0,
        "close_neurons": a.ref_wide - sca_m.neuron_corr if not np.isnan(a.ref_wide) else 0.0,
    }
    if spec_background(ds) and sca_m.angle_error > VERDICT_DENSE_ANGLE:
        drops["dense_background"] = max(drops["dense_background"], 1.0)
    cause, drop = max(drops.items(), key=lambda kv: kv[1])
    if drop > VERDICT_DROP:
        return "warning", cause, data
    best_other = max(ica_m.neuron_corr, sobi_m.neuron_corr)
    if ds.n_electrodes < k and sca_m.neuron_corr - best_other > VERDICT_WIN_MARGIN:
        return "success", "sca_wins", data
    if ds.n_electrodes >= k:
        return "info", "enough_electrodes", data
    return "info", "neutral", data
