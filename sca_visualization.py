"""Plotly-Visualisierungen der SCA-Demo: Elektrodenlayout, Signalspuren, Aktivität der Neuronen, Elektrodenvektoren (mit wahren und geschätzten Mischrichtungen), Richtungs-Histogramm,
Rekonstruktions-Beispiele, Zuordnungsmatrix, Kennzahlen-Balken, Sweeps und Szenen-Vergleich. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sca_constants as C
from sca_scenario import electrode_positions, neuron_position

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf"
NEURON_COLORS = ("#1f77b4", "#d68a2e", "#2ca02c", "#8e5fbf", "#c2185b")
BACKGROUND_COLORS = ("#7f7f7f",)
METHOD_COLORS = {"sca": BLUE, "single": GREEN, "ica": ORANGE, "sobi": PURPLE}
METHOD_NAMES = {"sca": "SCA (L1)", "single": "SCA (Einzelquelle)", "ica": "ICA", "sobi": "SOBI"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def source_color(i):
    return NEURON_COLORS[i] if i < len(NEURON_COLORS) else BACKGROUND_COLORS[0]


def source_labels(ds):
    labels = [f"Neuron {i + 1}" for i in range(ds.n_neurons)]
    for kind in ds.kinds[ds.n_neurons:]:
        labels.append("Gauß-Hintergrund" if kind == "gauss" else "Rhythmus")
    return labels


def build_layout(ds, spread=1.0):
    """Elektroden (Quadrate auf y = 0) und Neuronen (Kreise, Größe = Spitzenamplitude); ein Hintergrund wirkt flächig auf alle Elektroden."""
    pos = electrode_positions(ds.n_electrodes)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[f"E{j + 1}" for j in range(ds.n_electrodes)], textposition="bottom center", name="Elektroden",
                             marker=dict(symbol="square", size=14, color=GRAY), hoverinfo="skip"))
    for i in range(ds.n_neurons):
        x, y = neuron_position(i, spread)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"N{i + 1}"], textposition="top center", name=f"Neuron {i + 1}", hoverinfo="skip",
                                 marker=dict(size=10 + 14 * C.NEURON_AMPLITUDES[i], color=source_color(i), opacity=0.85)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-0.1, 1.1], title="Ort (willkürliche Einheit)", zeroline=False),
                      yaxis=dict(range=[-0.15, 0.7], title="Abstand", zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_traces(labels, arrays, t0, width, colors=None, spike_times=None, height=None, normalise=True):
    """Gestapelte Spuren eines Zeitfensters [t0, t0 + width) in ms (Abtastrate 10 kHz); optional Markierungen der wahren Spitzen je Zeile."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    fig = go.Figure()
    n = len(arrays)
    scale_all = max(float(np.abs(a).max()) for a in arrays) if not normalise else None
    for r, (label, y) in enumerate(zip(labels, arrays)):
        seg = y[lo:hi]
        scale = float(np.abs(y).max()) if normalise else scale_all
        offset = (n - 1 - r) * 1.3
        color = colors[r] if colors else BLUE
        fig.add_trace(go.Scatter(x=np.arange(lo, hi) * 1000.0 / fs, y=offset + seg / max(scale, 1e-12), mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="skip"))
        if spike_times is not None and r < len(spike_times):
            marks = [t for t in spike_times[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=[offset + 0.75] * len(marks), mode="markers", marker=dict(symbol="triangle-down", size=7, color=color), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height or max(180, 42 * n + 60), margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(title="Zeit [ms]"), yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=list(labels), zeroline=False))
    return lock_axes(fig)


def build_corr_heatmap(cm, row_labels, col_labels, title):
    """|Korrelation| wahre Quelle (Zeile) gegen Schätzung (Spalte): ein sauberes Bild hat in jeder Zeile und Spalte genau einen hellen Eintrag."""
    fig = go.Figure(go.Heatmap(z=cm, x=col_labels, y=row_labels, zmin=0, zmax=1, colorscale="Blues", text=np.round(cm, 2), texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=60 + 40 * len(row_labels), margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_activity(active, t0, width):
    """Wann welches Neuron aktiv ist (Wellenform über 10 % der Spitzenhöhe): eine Zeile je Neuron, unten die Zahl gleichzeitig aktiver Neuronen. Überlappung = Zahl über 1."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    m = active.shape[0]
    x = np.arange(lo, hi) * 1000.0 / fs
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.06)
    for i in range(m):
        y = active[i, lo:hi].astype(float)
        fig.add_trace(go.Scatter(x=x, y=(m - 1 - i) + 0.8 * y, mode="lines", line=dict(color=source_color(i), width=1.5), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=np.full(len(x), float(m - 1 - i)), mode="lines", line=dict(color=source_color(i), width=0.5), hoverinfo="skip"), row=1, col=1)
    count = active[:, lo:hi].sum(axis=0)
    fig.add_trace(go.Scatter(x=x, y=count, mode="lines", line=dict(color=RED if count.max() > 1 else GRAY, width=2, shape="hv"), fill="tozeroy", hoverinfo="skip"), row=2, col=1)
    fig.update_yaxes(tickmode="array", tickvals=[m - 1 - i + 0.4 for i in range(m)], ticktext=[f"Neuron {i + 1}" for i in range(m)], showgrid=False, row=1, col=1)
    fig.update_yaxes(title="aktiv", range=[0, max(3, float(active.sum(axis=0).max()) + 0.5)], dtick=1, row=2, col=1)
    fig.update_xaxes(title="Zeit [ms]", row=2, col=1)
    fig.update_layout(height=max(220, 40 * m + 120), margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return lock_axes(fig)


def _pca_axes(X, mask):
    V = (X - np.median(X, axis=1, keepdims=True))[:, mask]
    values, vectors = np.linalg.eigh(V @ V.T)
    return vectors[:, ::-1][:, :2]


def build_vectors(X, mask, A, A_hat, labels=None, colors=None):
    """Elektrodenvektoren x(t) der aktiven Zeitpunkte (bei mehr als zwei Elektroden auf die ersten zwei Hauptachsen projiziert) mit den wahren Mischrichtungen (grün, breit) und den geschätzten (rot, dünn).
    `mask` wählt die Zeitpunkte; mit `labels` (Cluster je Punkt von `mask`) werden sie nach Cluster gefärbt."""
    n = X.shape[0]
    Pm = np.eye(2) if n == 2 else _pca_axes(X, mask)
    V = (X - np.median(X, axis=1, keepdims=True))[:, mask]
    pts = Pm.T @ V
    fig = go.Figure()
    if labels is None:
        fig.add_trace(go.Scattergl(x=pts[0], y=pts[1], mode="markers", marker=dict(size=3, color=GRAY, opacity=0.4), hoverinfo="skip"))
    else:
        for j in np.unique(labels):
            sel = labels == j
            fig.add_trace(go.Scattergl(x=pts[0][sel], y=pts[1][sel], mode="markers", marker=dict(size=3, color=colors[int(j) % len(colors)], opacity=0.5), hoverinfo="skip"))
    lim = float(np.quantile(np.abs(pts), 0.999)) * 1.1
    for M, color, width, opacity in ((A, GREEN, 7, 0.5), (A_hat, RED, 2, 1.0)):
        for c in M.T:
            d = Pm.T @ (c / np.linalg.norm(c))
            if np.linalg.norm(d) < 1e-9:
                continue
            d = d / np.linalg.norm(d)
            fig.add_trace(go.Scatter(x=[-lim * d[0], lim * d[0]], y=[-lim * d[1], lim * d[1]], mode="lines", line=dict(color=color, width=width), opacity=opacity, hoverinfo="skip"))
    axis_title = ("Elektrode 1", "Elektrode 2") if n == 2 else ("Hauptachse 1", "Hauptachse 2")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis=dict(title=axis_title[0], range=[-lim, lim], scaleanchor="y", zeroline=True),
                      yaxis=dict(title=axis_title[1], range=[-lim, lim], zeroline=True))
    return lock_axes(fig)


def build_direction_histogram(U, A_hat):
    """Winkel jedes Elektrodenvektors zur nächsten geschätzten Mischachse (Grad): schmal = enge Cluster. Bei zwei Elektroden zusätzlich die Lage der geschätzten Achsen als senkrechte Linien im Winkel-Histogramm."""
    n = U.shape[1]
    fig = go.Figure()
    if n == 2:
        theta = np.degrees(np.arctan2(U[:, 1], U[:, 0])) % 180.0
        fig.add_trace(go.Histogram(x=theta, nbinsx=90, marker_color=BLUE, hoverinfo="skip"))
        for c in A_hat.T:
            fig.add_vline(x=float(np.degrees(np.arctan2(c[1], c[0])) % 180.0), line=dict(color=RED, width=1.5))
        fig.update_layout(xaxis=dict(title="Richtung des Elektrodenvektors [°] (Achse, 0-180)", range=[0, 180]), yaxis=dict(title="Anzahl Zeitpunkte"))
    else:
        cos = np.abs(U @ (A_hat / np.linalg.norm(A_hat, axis=0)))
        ang = np.degrees(np.arccos(np.clip(cos.max(axis=1), -1.0, 1.0)))
        fig.add_trace(go.Histogram(x=ang, nbinsx=60, marker_color=BLUE, hoverinfo="skip"))
        fig.update_layout(xaxis=dict(title="Winkel zur nächsten geschätzten Achse [°]"), yaxis=dict(title="Anzahl Zeitpunkte"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return lock_axes(fig)


def build_examples(labels, truth, l1, single, times):
    """Rekonstruktions-Beispiele: je ein Zeitpunkt (Spalte) die wahren Quellenwerte und die geschätzten (L1, Einzelquelle) - eine Gruppe je Zeitpunkt."""
    k = len(labels)
    fig = make_subplots(rows=1, cols=len(times), subplot_titles=[f"t = {t * 1000.0 / C.SAMPLE_RATE:.1f} ms" for t in times], horizontal_spacing=0.06)
    for c, t in enumerate(times, start=1):
        for values, name, color in ((truth[:, t], "wahr", GREEN), (l1[:, t], "SCA (L1)", BLUE), (single[:, t], "SCA (Einzelquelle)", ORANGE)):
            fig.add_trace(go.Bar(x=[f"N{i + 1}" for i in range(k)], y=values, name=name, marker_color=color, showlegend=(c == 1), hoverinfo="skip"), row=1, col=c)
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_method_bars(methods):
    """Neuronen-Korrelation und Spitzen-F1 der vier Verfahren."""
    fig = go.Figure()
    for name in ("sca", "single", "ica", "sobi"):
        m = methods[name]
        fig.add_trace(go.Bar(x=["Korrelation der Neuronen", "Spitzen-F1"], y=[m.neuron_corr, m.f1], name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name],
                             text=[f"{m.neuron_corr:.2f}", f"{m.f1:.2f}"], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_sweep(rows, xlabel, current=None, log=False):
    """Links: Neuronen-Korrelation der vier Verfahren (Mittel und Streuung über die Sweep-Datensätze); rechts: Winkelfehler der SCA-Mischspalten und Überlappungsanteil."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Korrelation der Neuronen", "Winkelfehler der SCA [°] und Überlappung"), horizontal_spacing=0.12, specs=[[{}, {"secondary_y": True}]])
    xs = [r["x"] for r in rows]
    for name in ("sca", "single", "ica", "sobi"):
        y = np.array([r[name] for r in rows])
        sd = np.array([r[name + "_std"] for r in rows])
        color = METHOD_COLORS[name]
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(y + sd) + list(y - sd)[::-1], fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=y, mode="lines+markers", name=METHOD_NAMES[name], line=dict(color=color, width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["angle"] for r in rows], mode="lines+markers", name="Winkelfehler [°]", line=dict(color=RED, width=2), hoverinfo="skip"), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=[r["overlap"] for r in rows], mode="lines+markers", name="Überlappung", line=dict(color=GRAY, width=2, dash="dot"), hoverinfo="skip"), row=1, col=2, secondary_y=True)
    fig.update_xaxes(title=xlabel, type="log" if log else "linear")
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title="Grad", rangemode="tozero", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title="Anteil", range=[0, 1], row=1, col=2, secondary_y=True)
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_scenes(rows):
    """Wer trennt was: Neuronen-Korrelation der vier Verfahren je Szene; Balken = Mittel, Fehlerbalken = Spanne über die Sweep-Datensätze."""
    labels = [r["scene"].replace(" (", "<br>(") for r in rows]
    fig = go.Figure()
    for name in ("sca", "single", "ica", "sobi"):
        y = [r[name] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip",
                             error_y=dict(type="data", symmetric=False, array=[r[name + "_max"] - r[name] for r in rows], arrayminus=[r[name] - r[name + "_min"] for r in rows])))
    fig.update_layout(height=420, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Korrelation der Neuronen", range=[0, 1.2]), legend=dict(orientation="h", y=-0.45))
    return lock_axes(fig)
