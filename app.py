"""Sparse Component Analysis (SCA) an Mehrelektroden-Signalen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - SCA - und lässt stattdessen das Beispiel wachsen.
Drittes Stück der Quellentrennung-Linie der "Konzepte"-Reihe: ein Nachfolger von ICA und SOBI, der den unterbestimmten Fall (mehr Neuronen als Elektroden) löst, indem er ausnutzt,
dass Neuronen selten gleichzeitig feuern. Was das bringt und wo es endet, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import sca_constants as C
from sca_algorithm import directions
from sca_evaluation import (
    SWEEP_LABELS, Settings, analyse_for, correlation_matrix, make_dataset, scene_table, sweep, verdict,
)
from sca_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from sca_visualization import (
    build_activity,
    build_corr_heatmap,
    build_direction_histogram,
    build_examples,
    build_layout,
    build_method_bars,
    build_scenes,
    build_sweep,
    build_traces,
    build_vectors,
    source_color,
    source_labels,
)

st.set_page_config(page_title="SCA – Sebastian Hanisch", layout="wide")

STEP_LABELS = {1: "1 · Quellen", 2: "2 · Mischung", 3: "3 · Elektrodenvektoren", 4: "4 · Clustering", 5: "5 · Rekonstruktion"}
WINDOW_WIDTH_MS = 60
SWEEP_OPTIONS = {"n_electrodes": "Anzahl Elektroden", "rate_scale": "Feuerrate", "noise": "Rauschen", "spread": "Neuronen-Abstand", "n_neurons": "Anzahl Neuronen"}


@st.cache_data(show_spinner=False)
def _dataset(m, n, g, kind, rate_scale, spread, noise, n_samples, seed):
    return make_dataset(m, n, g, kind, rate_scale, spread, noise, n_samples, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    m, n, g, kind, rate_scale, spread, noise, n_samples = base
    return sweep(parameter, settings=settings, m=m, n=n, g=g, kind=kind, rate_scale=rate_scale, spread=spread, noise=noise, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _scenes(base, settings):
    m, n, g, kind, rate_scale, spread, noise, n_samples = base
    return scene_table(settings, kind=kind, n_samples=n_samples)


st.title("🎯 Sparse Component Analysis – mehr Quellen als Elektroden")
st.markdown(
    """
Die **ICA** und **SOBI** aus den ersten beiden Stücken der Quellentrennung-Linie brauchen **mindestens so viele Elektroden wie Quellen**: Ohne das ist die Mischung nicht umkehrbar.
Bei einem Elektrodenarray kommt das ständig vor - man möchte fünf Neuronen mit drei Elektroden trennen. **SCA** (Sparse Component Analysis) nutzt eine andere Eigenschaft der Quellen:
**Neuronen feuern selten gleichzeitig.** Zu fast jedem Zeitpunkt ist höchstens **eine** Quelle aktiv - und dann zeigt der Elektrodenvektor $x(t)$ genau in die Richtung der **Mischspalte** dieses Neurons.
Die Richtungen fallen in **Cluster**, die Cluster verraten die Mischmatrix (auch wenn sie mehr Spalten als Zeilen hat), und danach lässt sich jeder Zeitpunkt in seine Quellen zerlegen -
so **dünn besetzt** wie möglich. Die Demo misst, was das bringt und wo es endet: hohe Feuerraten (die Quellen überlappen), Neuronen mit fast gleicher Richtung, Rauschen und eine einzige Elektrode.
Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - drittes Stück der Quellentrennung-Linie der \"Konzepte\"-Reihe, Nachfolger von ICA und SOBI - **ein** Verfahren "
    "an einem wachsenden Beispiel. Das Array ist dasselbe wie dort (ohne Laufzeitverzögerung); neu sind Feuerrate und Neuronen-Abstand als Regler."
)

with st.expander("So funktioniert SCA", expanded=True):
    st.markdown(
        """
**Das Modell** ist dasselbe wie bei ICA und SOBI: $x(t) = A\\,s(t) + \\text{Rauschen}$ mit $n$ Elektroden, $k$ Quellen und unbekannter Mischung $A \\in \\mathbb{R}^{n \\times k}$ - jetzt aber mit **$k > n$ erlaubt**.
Verlangt wird stattdessen **Dünnbesetztheit in der Zeit**: zu jedem Zeitpunkt sind nur wenige Quellen von null verschieden, meist eine.

1. **Aktive Zeitpunkte.** Nur Zeitpunkte, an denen der Elektrodenvektor deutlich aus dem Rauschen ragt, tragen Information. Die Norm $\\lVert x(t) \\rVert$ entscheidet (robuste Rauschschätzung aus den ruhigen Zeitpunkten).
2. **Richtungen.** Ist zum Zeitpunkt $t$ nur Neuron $i$ aktiv, gilt $x(t) = a_i\\,s_i(t)$: der Vektor $x(t)/\\lVert x(t) \\rVert$ zeigt (bis aufs Vorzeichen) in Richtung der Mischspalte $a_i$.
   Die Vorzeichen sind egal, weil die Spitzen positiv und negativ ausschlagen - es zählen **Achsen**, keine Pfeile.
3. **Clustering.** Die Richtungen fallen in $k$ Cluster. Ein **Achsen-k-Means** (Zuordnung nach $|u \\cdot c|$, Achse = Hauptvektor des Clusters, mehrere Neustarts) findet sie; die Achsen sind die geschätzten Mischspalten $\\hat A$.
   Nur **Kernpunkte** (Norm über 30 % der Spitzennorm) zählen: an Flanken und im Nachschlag der Spitzen liegen die Richtungen näher an Überlappung und Rauschen.
4. **Rekonstruktion.** Bei $k > n$ hat $\\hat A s = x$ unendlich viele Lösungen. Die **dünnste** - kleinste $\\ell_1$-Norm, $\\min \\lVert s \\rVert_1$ s.t. $\\hat A s = x$ - wird je Zeitpunkt per iterativ gewichteter Kleinste-Quadrate-Rechnung (IRLS) bestimmt;
   die Alternative ordnet jeden Zeitpunkt nur der Spalte mit dem besten Winkel zu (Einzelquelle).

Was SCA **verlangt**: Quellen, die selten gleichzeitig aktiv sind, und Mischrichtungen, die sich unterscheiden. Was es **nicht** verlangt: mindestens so viele Elektroden wie Quellen, Unabhängigkeit oder Nicht-Gaußianität.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_neurons = st.slider(
        "Neuronen", *bounds("n_neurons_slider"), key="n_neurons_slider",
        help="Spitzenartige Quellen wie in der ICA-Demo (fester Ort, feste Spitzenform, Feuerrate 20-35 Hz). Die Quellenzahl wird den Verfahren als bekannt vorgegeben.",
    )
    n_electrodes = st.slider(
        "Elektroden", *bounds("n_electrodes_slider"), key="n_electrodes_slider",
        help="Aufnahmestellen auf einer Zeile. Bei vier Neuronen: mit 2 Elektroden erreicht SCA die Neuronen-Korrelation 0.95, ICA 0.41 und SOBI 0.39; mit 3 Elektroden 0.98 / 0.68 / 0.68; ab 4 Elektroden liegt die ICA bei 0.98 - dann ist SCA nicht mehr nötig. "
             "Mit einer Elektrode gibt es keine Richtung: SCA findet nichts.",
    )
    rate_scale = st.slider(
        "Feuerrate (Faktor)", *bounds("rate_slider"), key="rate_slider", step=0.25,
        help="Faktor auf die Feuerraten (20-35 Hz). Mehr Feuern heißt mehr Überlappung: bei Faktor 1 sind an etwa 6 % der aktiven Zeitpunkte zwei Neuronen gleichzeitig aktiv, bei 4 an etwa 24 %. "
             "Bei 2 Elektroden fällt die Korrelation von 0.98 (Faktor 0.5) über 0.95 (1) und 0.91 (2) auf 0.83 (4).",
    )
    spread = st.slider(
        "Neuronen-Abstand", *bounds("spread_slider"), key="spread_slider", step=0.1,
        help="Abstand der Neuronen zur Mitte der Zeile (1 = wie in den anderen Demos). Klein = dicht beieinander = fast parallele Mischrichtungen. SCA findet die Richtungen trotzdem (Fehler unter 0.2 Grad), aber die L1-Rekonstruktion "
             "wird schlechter (bei 2 Elektroden und Abstand 0.1: 0.79; Einzelquelle 0.92).",
    )
    n_background = st.slider(
        "Hintergrundquellen", *bounds("n_background_slider"), key="n_background_slider",
        help="Eine zusätzliche flächige Quelle, die dauernd auf alle Elektroden wirkt - **nicht** sparse. Ein Gauß-Hintergrund senkt die Korrelation bei 3 Elektroden von 0.98 auf 0.96, ein Rhythmus auf 0.90.",
    )
    if n_background > 0:
        seed_widget("kind_select")
        kind = st.selectbox(
            "Art des Hintergrunds", C.BACKGROUND_KINDS, key="kind_select", format_func=lambda k: C.BACKGROUND_LABELS[k],
            help="Gauß-Rauschen (AR(1), Koeffizient 0.95) oder ein 10-Hz-Sinusrhythmus.",
        )
        st.session_state["_kind_kept"] = kind
    else:
        kind = st.session_state.get("_kind_kept", C.DEFAULT_BACKGROUND_KIND)
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Sensorrauschen relativ zum Neuronen-Signal. SCA ist robust: bei 3 Elektroden und 4 Neuronen Korrelation 0.99 (Rauschen 0), 0.93 (0.4), 0.74 (1.0); ICA 0.68 / 0.56 / 0.37.",
    )
    n_samples = st.slider(
        "Länge der Aufnahme", *bounds("n_samples_slider"), key="n_samples_slider", step=1000,
        help="Abtastwerte bei 10 kHz. Wenig Einfluss: bei 2 Elektroden und 4 Neuronen 0.94 mit 5000 und 0.95 mit 40000 Abtastwerten - die Cluster bilden sich schon aus wenigen Dutzend Spitzen.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**SCA**")
    reconstruction = st.selectbox(
        "Rekonstruktion", C.RECONSTRUCTIONS, key="reconstruction_select", format_func=lambda r: C.RECONSTRUCTION_LABELS[r],
        help="L1-Minimierung erlaubt mehrere aktive Quellen je Zeitpunkt und ist bei Überlappung besser; die Einzelquellen-Zuordnung ist bei fast parallelen Mischrichtungen besser (Abstand 0.1: 0.92 gegen 0.79).",
    )
    init_start = st.selectbox(
        "Start (Clustering und ICA)", C.INIT_STARTS, key="init_start_select", format_func=lambda i: f"Start {i}",
        help="Zufallsstart des Clusterings (10 Neustarts, das beste zählt) und der ICA. Auf diesen Daten ändert er kaum etwas: die Cluster sind klar.",
    )
    st.markdown("**ICA (zum Vergleich)**")
    contrast = st.selectbox(
        "Kontrastfunktion", C.CONTRASTS, key="contrast_select", format_func=lambda c: C.CONTRAST_LABELS[c],
        help="Wie die ICA Nicht-Gaußianität misst (siehe ICA-Demo). Für dieses Stück nebensächlich.",
    )

    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für Spikezeiten, Hintergrund und Rauschen.")

sync_query_params({
    "n_neurons_slider": int(n_neurons), "n_electrodes_slider": int(n_electrodes), "rate_slider": float(rate_scale), "spread_slider": float(spread), "n_background_slider": int(n_background),
    "kind_select": kind, "noise_slider": noise, "n_samples_slider": int(n_samples), "reconstruction_select": reconstruction, "contrast_select": contrast, "init_start_select": int(init_start), "seed_input": int(seed),
})

data_params = (int(n_neurons), int(n_electrodes), int(n_background), kind, float(rate_scale), float(round(spread, 2)), float(noise), int(n_samples), int(seed))
settings = Settings(reconstruction=reconstruction, contrast=contrast, init_start=int(init_start))
with st.spinner("Trenne die Quellen..."):
    ds = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
sca_model = analysis.models["sca"]
methods = analysis.methods
k = ds.S.shape[0]
m_n = ds.n_neurons
n_el = ds.n_electrodes
labels = source_labels(ds)
colors = [source_color(i) for i in range(k)]
level, code, vd = verdict(analysis)
data_key = data_params + (settings,)
sca_m, single_m, ica_m, sobi_m = methods["sca"], methods["single"], methods["ica"], methods["sobi"]

# --- SCA in Aktion -------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 SCA in Aktion")
if "sca_step" not in st.session_state or st.session_state.get("sca_step_owner") != data_key:
    st.session_state["sca_step"] = 1
    st.session_state["sca_step_owner"] = data_key
duration_ms = ds.S.shape[1] * 1000.0 / C.SAMPLE_RATE
max_start = int(duration_ms - WINDOW_WIDTH_MS)
if st.session_state.get("window_start", 0) > max_start:
    st.session_state["window_start"] = 0
step_col, play_col, win_col = st.columns([4, 2, 3])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="sca_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with win_col:
    window = st.slider(f"Zeitfenster ({WINDOW_WIDTH_MS} ms) ab [ms]", 0, max_start, key="window_start", step=10, help="Welchen Ausschnitt der Aufnahme die Signalspuren zeigen.")
peaks = ds.spike_times
view_slot = st.empty()
has_direction = n_el >= 2
if has_direction:
    U_core = directions(ds.X, sca_model.core)
n_active = ds.active.sum(axis=0)
example_times = []
for i in range(min(m_n, 3)):
    sel = [t for t in peaks[i] if 50 < t < ds.S.shape[1] - 50 and n_active[t] == 1]
    if sel:
        example_times.append(int(sel[len(sel) // 2]))
both = np.flatnonzero((n_active >= 2) & sca_model.mask)
if len(both):
    example_times.append(int(both[len(both) // 2]))


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die wahren Quellen** (in der Praxis unbekannt)")
            c1.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [peaks[i] if i < m_n else [] for i in range(k)]), width="stretch", key="step_sources")
            c2.markdown("**Wann welches Neuron aktiv ist**")
            c2.plotly_chart(build_activity(ds.active, window, WINDOW_WIDTH_MS), width="stretch", key="step_activity")
        elif current_step == 2:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Was die Elektroden messen**: jede Spur mischt die Quellen")
            c1.plotly_chart(build_traces([f"E{j + 1}" for j in range(n_el)], list(ds.X), window, WINDOW_WIDTH_MS, normalise=False), width="stretch", key="step_electrodes")
            c2.markdown("**Ort von Neuronen und Elektroden**")
            c2.plotly_chart(build_layout(ds, spread), width="stretch", key="step_layout")
        elif current_step == 3:
            if has_direction:
                st.markdown("**Elektrodenvektoren der aktiven Zeitpunkte** (grün, breit: wahre Mischrichtungen; rot: geschätzt)")
                st.plotly_chart(build_vectors(ds.X, sca_model.mask, ds.A, sca_model.A_hat), width="stretch", key="step_vectors")
            else:
                st.info("Mit nur einer Elektrode ist x(t) eine Zahl: es gibt keine Richtung, an der sich Neuronen unterscheiden ließen. SCA findet nichts (Korrelation 0).")
        elif current_step == 4:
            if has_direction:
                c1, c2 = st.columns([3, 2])
                c1.markdown("**Cluster der Kernpunkte** (Farbe = Cluster; rot: geschätzte Achsen)")
                c1.plotly_chart(build_vectors(ds.X, sca_model.core, ds.A, sca_model.A_hat, sca_model.clustering.labels, colors), width="stretch", key="step_clusters")
                c2.markdown("**Wie eng die Cluster sind**")
                c2.plotly_chart(build_direction_histogram(U_core, sca_model.A_hat), width="stretch", key="step_histogram")
            else:
                st.info("Ohne Richtung gibt es nichts zu clustern.")
        else:
            if has_direction and example_times:
                st.markdown("**Ausgewählte Zeitpunkte: wahre Quellenwerte und Rekonstruktion** (Skala und Vorzeichen der wahren Quellen angepasst; die letzten Beispiele sind Zeitpunkte mit zwei aktiven Neuronen, falls vorhanden)")
                st.plotly_chart(build_examples(labels[:m_n], ds.S[:m_n], sca_m.aligned[:m_n], single_m.aligned[:m_n], example_times), width="stretch", key="step_examples")
            elif not has_direction:
                st.info("Ohne Richtung gibt es nichts zu rekonstruieren.")
            c1, c2 = st.columns(2)
            c1.markdown("**Wahre Quellen**")
            c1.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors), width="stretch", key="step_result_true")
            c2.markdown("**SCA-Schätzungen**")
            c2.plotly_chart(build_traces(labels, list(sca_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="step_result_sca")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"Die Neuronen feuern selten: an {analysis.active_fraction:.0%} der Zeitpunkte ist mindestens ein Neuron aktiv (Wellenform über 10 % der Spitzenhöhe), und nur an {analysis.overlap_fraction:.0%} davon sind es zwei oder mehr. "
               "Das ist die Annahme von SCA: fast jeder aktive Zeitpunkt gehört genau einem Neuron. Mit dem Regler 'Feuerrate' wächst die Überlappung.")
elif step == 2:
    st.caption(f"{n_el} Elektrode(n) messen jeweils eine gewichtete Summe der {k} Quellen plus Rauschen ({noise:.2f} der Stärke des Neuronen-Signals). "
               f"{'Mehr Quellen als Elektroden (' + str(k) + ' > ' + str(n_el) + '): die Mischung ist nicht umkehrbar - ICA und SOBI können hier nichts ausrichten.' if k > n_el else 'Genug Elektroden: hier reichen auch ICA und SOBI.'}")
elif step == 3:
    st.caption("Jeder Punkt ist ein Zeitpunkt. Zu Spitzenzeitpunkten eines einzelnen Neurons liegt der Punkt auf der Achse seiner Mischspalte - die Punktwolke bildet **Strahlen** (bei mehr als zwei Elektroden auf die ersten beiden Hauptachsen projiziert; "
               "die Strahlen der Neuronen überlagern sich dann teilweise). Punkte zwischen den Strahlen sind Überlappungen und Rauschen.")
elif step == 4:
    st.caption(f"Der Achsen-k-Means fasst die Kernpunkte in {k} Cluster; die Achsen der Cluster sind die geschätzten Mischspalten. Mittlerer Winkelfehler zur wahren Mischspalte: **{sca_m.angle_error:.1f}°** "
               f"(Zufallsniveau bei zufälligen Achsen: {analysis.chance_angle:.0f}°). Kleinster Winkel zwischen zwei wahren Spalten: {analysis.min_pair_angle:.1f}°.")
else:
    st.caption("Bei mehr Quellen als Elektroden gibt es zu jedem Zeitpunkt unendlich viele Zerlegungen; L1 wählt die dünnste. An Zeitpunkten mit nur einem aktiven Neuron trifft sie es exakt, bei zwei aktiven Neuronen "
               "nur, wenn deren Mischrichtungen verschieden genug sind. Die Einzelquellen-Zuordnung lässt jedem Zeitpunkt nur ein Neuron.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wer trennt was: SCA, ICA und SOBI auf denselben Daten")
st.caption(
    "Neuronen-Korrelation: mittlere |Korrelation| der Neuronen mit der zugeordneten Schätzung (ICA und SOBI liefern bei weniger Elektroden als Quellen nur so viele Komponenten wie Elektroden). "
    "Winkelfehler: mittlerer Winkel zwischen wahrer und geschätzter Mischspalte (nur SCA schätzt sie), zum Vergleich das Zufallsniveau."
)
m1, m2, m3, m4 = st.columns(4)
best_other = max(ica_m.neuron_corr, sobi_m.neuron_corr)
m1.metric("Neuronen-Korrelation (SCA)", f"{sca_m.neuron_corr:.2f}", delta=f"{sca_m.neuron_corr - best_other:+.2f} ggü. bester von ICA/SOBI", delta_color="normal", help="Mittlere |Korrelation| der Neuronen mit ihrer Schätzung. 1 = perfekt.")
m2.metric("Winkelfehler der Mischspalten", f"{sca_m.angle_error:.1f}°" if n_el >= 2 else "-", delta=f"Zufall: {analysis.chance_angle:.0f}°" if n_el >= 2 else None, delta_color="off",
          help="Mittlerer Winkel (Grad) zwischen den wahren und den von SCA geschätzten Mischspalten der Neuronen, Achsen ohne Vorzeichen.")
m3.metric("Spitzen-F1 (SCA)", f"{sca_m.f1:.2f}", delta=f"{sca_m.f1 - ica_m.f1:+.2f} ggü. ICA", delta_color="normal", help="Wie viele der wahren Spitzen auf der Schätzung gefunden werden (Toleranz ±4 Abtastwerte).")
m4.metric("Überlappung", f"{analysis.overlap_fraction:.0%}", help="Anteil der aktiven Zeitpunkte, an denen mindestens zwei Neuronen gleichzeitig aktiv sind. Je größer, desto weiter weg von der SCA-Annahme.")

if code == "one_electrode":
    st.warning(f"⚠️ Mit einer Elektrode gibt es keine Richtung: SCA findet nichts (Korrelation {vd['sca']:.2f}); auch ICA ({vd['ica']:.2f}) und SOBI ({vd['sobi']:.2f}) haben mit einer Komponente für {k} Quellen keine Chance.")
elif code == "dense_background":
    st.warning(f"⚠️ Der Hintergrund wirkt dauernd auf alle Elektroden - er ist **nicht sparse**. Zu jedem Zeitpunkt ist er zusätzlich zum Neuron aktiv, die Richtungen verschieben sich: Winkelfehler {vd['angle']:.1f}° "
               f"(ohne Hintergrund unter 1.2°), Korrelation {vd['sca']:.2f} statt {vd['ref_no_background']:.2f}. ICA {vd['ica']:.2f}, SOBI {vd['sobi']:.2f}.")
elif code == "noise":
    st.warning(f"⚠️ Das Rauschen kostet viel: Neuronen-Korrelation {vd['sca']:.2f} statt {vd['ref_clean']:.2f} ohne Rauschen (Richtungen noch auf {vd['angle']:.1f}° genau). ICA {vd['ica']:.2f} und SOBI {vd['sobi']:.2f} leiden mehr.")
elif code == "overlap_high":
    st.warning(f"⚠️ An {vd['overlap']:.0%} der aktiven Zeitpunkte sind mindestens zwei Neuronen gleichzeitig aktiv: die Annahme 'höchstens eine aktive Quelle' wird brüchig. Die Richtungen bleiben gut gefunden ({vd['angle']:.1f}°), "
               f"die Rekonstruktion leidet: Korrelation {vd['sca']:.2f} statt {vd['ref_low_rate']:.2f} bei einem Viertel der Feuerrate.")
elif code == "close_neurons":
    st.warning(f"⚠️ Die Mischrichtungen der Neuronen sind fast parallel (kleinster Winkel {vd['min_angle']:.1f}°). Die Richtungen findet SCA trotzdem ({vd['angle']:.1f}° Fehler), aber die Zerlegung eines Zeitpunkts in fast parallele Spalten ist schlecht bestimmt: "
               f"Korrelation {vd['sca']:.2f} statt {vd['ref_wide']:.2f} bei weit auseinanderliegenden Neuronen. "
               f"{'Die Einzelquellen-Zuordnung ist hier besser (' + format(vd['single'], '.2f') + ').' if vd['single'] > vd['sca'] else ''}")
elif code == "sca_wins":
    st.success(f"✅ SCA trennt die Neuronen (Korrelation {vd['sca']:.2f}), obwohl es mit {k} Quellen und {n_el} Elektroden mehr Quellen als Elektroden gibt - ICA ({vd['ica']:.2f}) und SOBI ({vd['sobi']:.2f}) können das nicht. "
               f"Die Mischrichtungen sind auf {vd['angle']:.1f}° genau gefunden (Zufallsniveau {analysis.chance_angle:.0f}°). Spitzen-F1 {vd['f1']:.2f} gegen {vd['ica_f1']:.2f} (ICA).")
elif code == "enough_electrodes":
    st.info(f"ℹ️ Mit {n_el} Elektroden für {k} Quellen ist die Mischung umkehrbar: die ICA reicht ({vd['ica']:.2f}), SOBI ebenso ({vd['sobi']:.2f}). SCA ({vd['sca']:.2f}) ist gleichauf oder besser, weil es Rauschen an ruhigen "
            "Zeitpunkten weglässt - hier aber nicht nötig.")
else:
    st.info(f"ℹ️ Neuronen-Korrelation: SCA {vd['sca']:.2f}, ICA {vd['ica']:.2f}, SOBI {vd['sobi']:.2f}.")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Wahre Quellen**")
    st.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [peaks[i] if i < m_n else [] for i in range(k)]), width="stretch", key="res_true")
with t2:
    st.markdown("**SCA**")
    st.plotly_chart(build_traces(labels, list(sca_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_sca")
t3, t4 = st.columns(2)
with t3:
    st.markdown("**ICA**")
    st.plotly_chart(build_traces(labels, list(ica_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_ica")
with t4:
    st.markdown("**SOBI**")
    st.plotly_chart(build_traces(labels, list(sobi_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_sobi")

h1, h2, h3 = st.columns(3)
with h1:
    st.plotly_chart(build_corr_heatmap(correlation_matrix(ds.S, sca_m.estimates), labels, [f"K{j + 1}" for j in range(sca_m.estimates.shape[0])], "SCA: |Korrelation|"), width="stretch", key="heat_sca")
with h2:
    st.plotly_chart(build_corr_heatmap(correlation_matrix(ds.S, ica_m.estimates), labels, [f"IC{j + 1}" for j in range(ica_m.estimates.shape[0])], "ICA: |Korrelation|"), width="stretch", key="heat_ica")
with h3:
    st.plotly_chart(build_method_bars(methods), width="stretch", key="method_bars")
st.caption("Links (Zeilen = wahre Quellen, Spalten = Komponenten): eine saubere Trennung hat in jeder Zeile und Spalte genau einen hellen Eintrag. SCA liefert immer so viele Komponenten wie Quellen, ICA nur so viele wie Elektroden. "
           "Rechts: Korrelation der Neuronen und Spitzen-F1 der vier Verfahren.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Elektroden, Feuerrate, Rauschen und Neuronen-Abstand ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_OPTIONS), format_func=lambda p: SWEEP_OPTIONS[p], key="sweep_select")
current = {"n_electrodes": int(n_electrodes), "rate_scale": float(rate_scale), "noise": float(noise), "spread": float(spread), "n_neurons": int(n_neurons)}[sweep_param]
with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
    rows = _sweep(sweep_param, data_params[:8], settings)
st.plotly_chart(build_sweep(rows, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
st.caption("Mittel und Streuung über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Rechts: Winkelfehler der SCA-Mischspalten (rot) und Überlappungsanteil (grau, gepunktet).")

st.markdown("---")

# --- Szenen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🧩 Wer trennt was: sieben Szenen im Vergleich")
if st.button("Sieben Szenen vergleichen (dauert einige Sekunden)", key="scenes_start"):
    st.session_state["scenes_on"] = True
if st.session_state.get("scenes_on"):
    with st.spinner("Vergleiche 7 Szenen × 5 Datensätze × 4 Verfahren..."):
        scene_rows = _scenes(data_params[:8], settings)
    st.plotly_chart(build_scenes(scene_rows), width="stretch", key="scenes_chart")
    st.table({
        "Szene": [r["scene"] for r in scene_rows],
        "SCA (L1)": [f"{r['sca']:.2f} ({r['sca_min']:.2f}-{r['sca_max']:.2f})" for r in scene_rows],
        "SCA (Einzelquelle)": [f"{r['single']:.2f}" for r in scene_rows],
        "ICA": [f"{r['ica']:.2f} ({r['ica_min']:.2f}-{r['ica_max']:.2f})" for r in scene_rows],
        "SOBI": [f"{r['sobi']:.2f}" for r in scene_rows],
        "Winkelfehler SCA": ["-" if np.isnan(r["angle"]) else f"{r['angle']:.1f}°" for r in scene_rows],
    })
    st.caption("Jede Szene legt Neuronen- und Elektrodenzahl und ihre Besonderheit fest; Rauschen (sofern die Szene es nicht setzt), Länge und Rekonstruktions-Einstellungen wie in der Seitenleiste. Mittel und Spanne über 5 feste Datensätze.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden - und wer danach kommt")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **selten gleichzeitig aktiv** | Hohe Feuerraten: die Rekonstruktion leidet (Preset "Hohe Feuerrate"), die Richtungen bleiben lange gut. Dauerhaft aktive Quellen (Hintergrund) verschieben die Richtungen (Preset "Dichter Rhythmus-Hintergrund"). | Zeit-Frequenz-Masken (andere Sparsität), **ICA/SOBI** bei genug Elektroden |
| **verschiedene Mischrichtungen** | Fast parallele Spalten: Richtungen gefunden, Zerlegung schlecht bestimmt (Preset "Neuronen dicht beieinander"). | Zusätzliche Information: Wellenform, zeitliche Struktur (Spike-Sorting-Zweig) |
| **mindestens zwei Elektroden** | Eine Elektrode: keine Richtung, SCA findet nichts. | Wellenform-basierte Verfahren (Spike-Sorting-Zweig) |
| **Quellenzahl bekannt** | Zu wenige Cluster verschmelzen Neuronen, zu viele teilen sie. | Modellwahl (nicht in dieser Demo) |
| **Vorzeichen frei / Quellen beliebig** | Bei nur positiven Quellen (Leistung, Verbrauch) ist das Vorzeichen nicht beliebig. | **NMF**: Nichtnegativität statt Sparsität |
| **momentane Mischung** | Laufzeitunterschiede zwischen den Elektroden (hier ausgeblendet, siehe ICA-Demo) verschieben die Richtungen je Elektrode. | **Spike-Sorting-Zweig** inkl. Verzögerungsgraph |
"""
)
st.caption("Die genannten Verfahren sind die nächsten Stücke der Quellentrennung-Linie; hier steht nur, welche Annahme sie jeweils lockern.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $x(t) = A\,s(t) + \varepsilon(t)$, $x \in \mathbb{R}^n$, $s \in \mathbb{R}^k$, $A \in \mathbb{R}^{n \times k}$, $k \ge n$ erlaubt. **Sparsität in der Zeit:** für die meisten $t$ hat $s(t)$ höchstens eine von null verschiedene Komponente
(allgemein höchstens $n - 1$; dann ist die Zerlegung eindeutig, wenn je $n$ Spalten von $A$ linear unabhängig sind).
Identifizierbar nur bis auf Reihenfolge, Vorzeichen und Skala der Spalten (Georgiev et al.; Bofill & Zibulevsky für den Clustering-Ansatz).

**Aktive Zeitpunkte.** $r(t) = \lVert x(t) - \tilde m \rVert$ mit dem Median $\tilde m$ je Elektrode; aktiv, wenn $r(t) > \max(4\,q_{0.2},\ 0.1\,q_{0.995})$ ($q_p$ = $p$-Quantil von $r$).
**Kernpunkte** für das Clustering: aktiv und $r(t) > 0.3\,q_{0.995}$.

**Achsen-k-Means.** Für $u_t = x(t)/\lVert x(t) \rVert$: Zuordnung $z_t = \arg\max_j |u_t^\top c_j|$; Achse $c_j$ = Eigenvektor zum größten Eigenwert von $\sum_{t: z_t = j} u_t u_t^\top$ (Vorzeichen egal);
Start per $k$-means++-Auswahl auf Achsen mit $d = 1 - \max_j |u^\top c_j|$; 10 Neustarts, bester Zielwert $\tfrac1N \sum_t \max_j |u_t^\top c_j|$. $\hat A$ = Achsen als Spalten, Vorzeichen so, dass die Spaltensumme nicht negativ ist.

**$\ell_1$-Rekonstruktion.** $\min_s \lVert s \rVert_1$ s.t. $\hat A s = x(t)$, gelöst durch **IRLS** (FOCUSS): $s^{(i+1)} = W\hat A^\top (\hat A W \hat A^\top + \sigma^2 I)^{-1} x$ mit $W = \operatorname{diag}(|s^{(i)}| + \epsilon_i)$, Start
$s^{(0)} = \hat A^+ x$, $\epsilon_i \to 0$, Regularisierung $\sigma^2$ = Rauschvarianz der ruhigen Zeitpunkte (30 Iterationen, vektorisiert über alle Zeitpunkte). **Einzelquelle:** $s_j = \hat a_j^\top x$ für $j = \arg\max_j |\hat a_j^\top x|/\lVert x \rVert$, alle anderen 0.
An inaktiven Zeitpunkten sind alle Schätzungen 0.

**Kennzahlen.** Zuordnung und Regression wie in der ICA-Demo. **Winkelfehler:** $\arccos |\hat a_j^\top a_i| / (\lVert \hat a_j \rVert \lVert a_i \rVert)$ nach optimaler Zuordnung; **Zufallsniveau:** derselbe Fehler für zufällige Achsen im positiven Orthanten (300 Ziehungen).
**Überlappung:** Anteil der Zeitpunkte mit mindestens zwei aktiven Neuronen unter denen mit mindestens einem (Neuron aktiv: Wellenform über 10 % der Spitzenhöhe).

**Grenzen.** (1) Dichte Quellen (Hintergrund) verletzen die Sparsität an jedem Zeitpunkt. (2) Hohe Überlappung: der Zeitpunkt gehört zwei Spalten, L1 findet sie nur bei ausreichend verschiedenen Richtungen. (3) Fast parallele Spalten: schlecht konditionierte Zerlegung.
(4) Eine Elektrode: keine Richtung. (5) Die **Quellenzahl** wird als bekannt angenommen. (6) Die Zerlegung ist je Zeitpunkt unabhängig - zeitliche Struktur bleibt ungenutzt.

Implementiert in `sca_algorithm.py` (aktive Zeitpunkte, Achsen-k-Means, L1/IRLS, Einzelquelle), `sca_ica.py` und `sca_sobi.py` (Vergleichsverfahren, wortgleich aus ica-demo/sobi-demo), `sca_scenario.py` (Mehrelektroden-Generator),
`sca_evaluation.py` (Zuordnung, Kennzahlen, Sweeps, Szenen, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
