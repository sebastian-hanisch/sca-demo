# Sparse Component Analysis (SCA) – mehr Quellen als Elektroden – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-sca-demo.streamlit.app/)**

Drittes Stück der **Quellentrennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Sparse Component Analysis** – an einem wachsenden Beispiel, mit **ICA** und **SOBI** als Vergleich.
Vehikel: dasselbe **Mehrelektroden-Array** wie in [ica-demo](../ica-demo) und [sobi-demo](../sobi-demo) (Neuronen, Wellenformen, Feuerraten und Mischung dort übernommen, per Test gegen eingefrorene Werte geprüft), ohne Laufzeitverzögerung;
neu sind die Regler **Feuerrate** (steuert, wie oft zwei Neuronen gleichzeitig aktiv sind) und **Neuronen-Abstand** (steuert, wie ähnlich die Mischrichtungen sind) sowie ein optionaler **dauerhaft aktiver Hintergrund**.

**Einordnung in die Reihe (die Kanten des Graphen):** SCA löst den **unterbestimmten Fall (mehr Quellen als Elektroden)**, an dem ICA und SOBI scheitern. Statt Unabhängigkeit oder Autokorrelation nutzt es **Sparsität in der Zeit**: Neuronen feuern selten gleichzeitig, also zeigt der
Elektrodenvektor x(t) zu fast jedem Zeitpunkt in die Richtung der Mischspalte **eines** Neurons. Die Kante ist ein **Ast direkt nach der ICA** (und nach SOBI: dieselbe Schwäche); der Richtungs-Clustering-Ansatz steht dem Spike-Sorting-Zweig der Linie nahe.
```
ica-demo → sobi-demo
         → sca-demo           (Sparsität statt Unabhängigkeit: mehr Quellen als Sensoren)
         → NMF                (Nichtnegativität statt Unabhängigkeit)
         → Spike-Sorting-Zweig (Standard-Pipeline, Vorlagenabgleich, Verzögerungsgraph: spike-sorting-demo, template-matching-demo, delay-graph-demo)
```
Gebaut ist inzwischen der Spike-Sorting-Zweig; NMF fehlt noch. Die Demo selbst markiert nur, welche Annahme die Nachfolger jeweils lockern.

| Frage | Ergebnis (Neuronen-Korrelation, 4 Neuronen, Rauschen 0.05, 20000 Abtastwerte; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Weniger Elektroden als Neuronen | ✅ **2 Elektroden:** SCA **0.95**, ICA 0.41, SOBI 0.39 (Einzelquelle 0.95); **3 Elektroden:** 0.98 / 0.68 / 0.68. Mischrichtungen auf **0.6°** (2 El.) bzw. 0.5° genau, Zufallsniveau 15° bzw. 23° |
| Fünf Neuronen | ✅ 2 Elektroden: SCA 0.90 (Einzelquelle 0.91), ICA 0.30; 3 Elektroden: **0.97** gegen 0.52; 4: 0.98 gegen 0.75; 5: 0.98 gegen 0.82 |
| Genug Elektroden (6) | ✅ ICA 0.981 und SOBI 0.978 reichen; SCA 0.995 ist trotzdem gleichauf oder besser (lässt Rauschen an ruhigen Zeitpunkten weg) – wird hier aber nicht gebraucht |
| Eine Elektrode | ❌ 0.00 – es gibt keine Richtung (ICA/SOBI 0.16 mit einer Komponente für vier Neuronen) |
| Feuerrate (2 Elektroden) | ⚠️ Faktor 0.5: 0.98, 1: 0.95, 2: 0.91, **4: 0.83**; Überlappung (Anteil der aktiven Zeitpunkte mit ≥ 2 aktiven Neuronen) 3 % / 6 % / 13 % / 24 %; Richtungsfehler wächst nur von 0.3° auf 2.1° |
| Neuronen dicht beieinander (2 Elektroden, Abstand 0.1) | ⚠️ Richtungen auf **0.1°** genau, aber L1-Rekonstruktion 0.79 (Einzelquelle **0.92**) – fast parallele Spalten machen die Zerlegung eines Zeitpunkts schlecht bestimmt |
| Dauernd aktiver Hintergrund (3 Elektroden) | ⚠️ ohne 0.985, Gauß-Hintergrund 0.96, **Rhythmus 0.90** (Richtungsfehler 3.3–5.0° statt unter 1.2°); ICA 0.60, SOBI 0.37 |
| Rauschen (3 Elektroden) | ⚠️ 0 → 0.99, 0.2 → 0.97, 0.4 → 0.93, 0.7 → 0.84, **1.0 → 0.74**; ICA 0.68 / 0.64 / 0.56 / 0.45 / 0.37 – SCA leidet weniger |
| Länge der Aufnahme | ✅ kaum relevant: 0.94 mit 5000, 0.95 mit 40000 Abtastwerten (2 Elektroden) – die Cluster bilden sich aus wenigen Dutzend Spitzen |
| Rechenzeit | ✅ ≈ 0.1 s (T = 20000, 4 Neuronen); 0.44 s für die gesamte Analyse mit vier Verfahren bei T = 40000, 5 Neuronen, 8 Elektroden und Hintergrund |

## Was die Demo zeigt

1. **SCA in Aktion** (Schritt-Slider + Abspielen, Zeitfenster-Regler): **Quellen** (Spuren und *wann welches Neuron aktiv ist*, mit Zahl gleichzeitig aktiver Neuronen) → **Mischung** → **Elektrodenvektoren** (die aktiven Zeitpunkte als Punktwolke: **Strahlen** in Richtung der
   Mischspalten; grün breit = wahre, rot = geschätzte Richtungen; bei mehr als zwei Elektroden auf die ersten zwei Hauptachsen projiziert) → **Clustering** (Cluster-Farben, Histogramm der Richtungen bzw. des Winkels zur nächsten Achse) →
   **Rekonstruktion** (ausgewählte Zeitpunkte: wahre Quellenwerte gegen L1 und Einzelquelle, auch Zeitpunkte mit zwei aktiven Neuronen; wahre gegen geschätzte Spuren).
2. **Wer trennt was: SCA, ICA und SOBI auf denselben Daten:** Neuronen-Korrelation, Winkelfehler der Mischspalten mit **berechnetem Zufallsniveau**, Spitzen-F1, Überlappung; die vier Spurgruppen; Zuordnungsmatrizen; Urteil
   (Codes: eine Elektrode → dauernd aktiver Hintergrund → Rauschen → hohe Überlappung → Neuronen dicht beieinander → SCA vorn → genug Elektroden → neutral; bei mehreren Ursachen gewinnt die mit dem größten Verlust, gemessen an Referenzläufen).
3. **📐 Sweeps** über Elektrodenzahl, Feuerrate, Rauschen, Neuronen-Abstand und Neuronenzahl (feste Datensätze ab 100000, Streuung, aktueller Wert markiert; rechts Winkelfehler und Überlappung).
4. **🧩 Wer trennt was:** sieben Szenen (genug Elektroden, zwei Elektroden, fünf Neuronen mit drei, hohe Feuerrate, dicht beieinander, Rhythmus-Hintergrund, starkes Rauschen) für SCA (L1 und Einzelquelle), ICA und SOBI mit Spanne über die Datensätze (Experiment auf Abruf).
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Zeit-Frequenz-Masken, ICA/SOBI, Spike-Sorting-Zweig, NMF).

Regler: Neuronen (2–5), Elektroden (1–8), Feuerrate (×0.25–×4), Neuronen-Abstand (0.1–1), Hintergrund (0/1, Art bei 0 ausgeblendet, Auswahl bleibt erhalten), Rauschen, Länge, Rekonstruktion (L1 / Einzelquelle), Start (Clustering und ICA), ICA-Kontrast.

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | SCA | Einzelquelle | ICA | SOBI | Richtungsfehler | Urteil |
|---|---|---|---|---|---|---|
| Vier Neuronen, zwei Elektroden | 0.952 | 0.947 | 0.406 | 0.394 | 1.1° | SCA vorn |
| Genug Elektroden (6) | 0.994 | 0.953 | 0.981 | 0.978 | 0.9° | genug Elektroden |
| Hohe Feuerrate (×4) | 0.829 | 0.782 | 0.408 | 0.377 | 1.8° | Überlappung hoch (26 %) |
| Neuronen dicht beieinander (0.1) | 0.792 | **0.911** | 0.375 | 0.373 | 0.1° | Neuronen dicht beieinander |
| Dichter Rhythmus-Hintergrund (3 El.) | 0.899 | 0.843 | 0.597 | 0.365 | 3.5° | dauernd aktiver Hintergrund |
| Starkes Rauschen (1.0, 3 El.) | 0.736 | 0.720 | 0.377 | 0.351 | 2.1° | Rauschen |

## Modell und Verfahren

- **Szenario** (`sca_scenario.py`): wie in ica-demo/sobi-demo (10 kHz; Neuronen mit biphasischer Wellenform, Poisson-artiges Feuern mit 2 ms Refraktärzeit, Mischung ∝ 1/(d² + ε), Rauschen relativ zum Neuronen-Signal, eigener Zufallsstrom je Quelle); neu: Feuerraten-Faktor, Neuronen-Abstand
  (Abstand der Neuronen zur Mitte der Zeile), Aktivität je Neuron und Zeitpunkt (Wellenform über 10 % der Spitzenhöhe), optionaler Hintergrund (AR(1) 0.95 oder 10-Hz-Sinus, gleich stark auf allen Elektroden).
- **SCA** (`sca_algorithm.py`, numpy von Grund auf): **aktive Zeitpunkte** (Norm des Elektrodenvektors über max(4 × Rausch-Norm, 10 % der Spitzen-Norm)); **Kernpunkte** (zusätzlich über 30 % der Spitzen-Norm); **Achsen-k-Means** (Zuordnung nach |u·c|, Achse = Hauptvektor der Streumatrix, k-means++-Start auf Achsen, 10 Neustarts,
  Vorzeichen egal, kanonisches Vorzeichen: Spaltensumme ≥ 0); **Rekonstruktion** je aktivem Zeitpunkt per **L1-Minimierung** (IRLS/FOCUSS, vektorisiert, mit Rausch-Regularisierung aus den ruhigen Zeitpunkten) oder **Einzelquelle** (beste Spalte).
- **Vergleich:** ICA und SOBI wortgleich aus ica-demo/sobi-demo (`sca_ica.py`, `sca_sobi.py`, gegen eingefrorene Werte geprüft); bei weniger Elektroden als Quellen liefern sie nur so viele Komponenten wie Elektroden.
- **Auswertung** (`sca_evaluation.py`): Zuordnung (Bitmasken-DP), Neuronen-Korrelation, Spitzen-F1 (aus ica-demo); neu **Winkelfehler der Mischspalten** nach optimaler Zuordnung mit **simuliertem Zufallsniveau** (zufällige Achsen im positiven Orthanten), Überlappungsanteil, kleinster Winkel zwischen wahren Spalten,
  Sweeps, Szenen-Tabelle, Urteil mit Referenzläufen (ohne Rauschen, bei einem Viertel der Feuerrate, bei Neuronen-Abstand 1, ohne Hintergrund).

## Was nicht funktioniert hat / Grenzen

- **Erste Fassung mit allen aktiven Zeitpunkten war unzuverlässig:** in 6 von 24 Fällen (5 Neuronen, 2/3/5 Elektroden, 8 Datensätze) verschmolz das Clustering benachbarte Neuronen (Mischspalten-Fehler bis 57°), obwohl der Zielwert des Clusterings fast gleich blieb (0.993 gegen 0.994) –
  er unterscheidet gute und schlechte Lösungen kaum, deshalb helfen auch mehr Neustarts nicht. Erst die **Beschränkung auf Kernpunkte** (Norm über 30 % der Spitzen-Norm) machte es zuverlässig (0 von 24). Ein Energie-Gewicht half nur halb (weiterhin 1–2 von 8 Fällen mit mehr als 3° Fehler).
- **Die L1-Rekonstruktion ist nicht immer die beste:** bei fast parallelen Mischspalten (Neuronen dicht beieinander, 2 Elektroden) verteilt sie Energie auf mehrere Spalten (0.79), die Einzelquellen-Zuordnung liegt bei 0.92. Bei zwei aktiven Neuronen ist die L1-Lösung nur dann die wahre, wenn die Richtungen verschieden genug sind
  (im Test gibt es Instanzen, in denen selbst die exakte LP-Lösung nicht die wahre ist). Bei hoher Überlappung (Faktor 4, 2 Elektroden) liegt L1 mit 0.83 knapp vor der Einzelquelle (0.78).
- **Die IRLS-Iteration konvergiert nur langsam gegen die exakte L1-Lösung** (gegen `scipy.optimize.linprog` geprüft: 150 Iterationen mit Abklingfaktor 0.85); die Demo verwendet 30. 60–200 Iterationen verändern die Neuronen-Korrelation um höchstens 0.002.
- **SCA schlägt die ICA auch mit genug Elektroden** (0.995 gegen 0.981): das ist kein Argument gegen die ICA, sondern ein Effekt der Szene (Neuronen sind sparse, das Rauschen wird an ruhigen Zeitpunkten weggelassen). Bei nicht-sparsen Quellen wäre es umgekehrt.
- **Der Winkelfehler sagt wenig über die Rekonstruktion:** bei Neuronen dicht beieinander bleibt er unter 0.2°, die Korrelation fällt trotzdem auf 0.79; beim Rhythmus-Hintergrund (3.5°) fällt sie auf 0.90.
- **Rauschen + dicht beieinander** verstärken sich: bei Abstand 0.1, 3 Elektroden und Rauschen 0.4 fällt die Korrelation auf 0.60 (Richtungsfehler 3.1°), bei Rauschen 0.7 auf 0.48. Das Urteil nennt hier das Rauschen als Ursache, weil sich die beiden Effekte nicht sauber trennen lassen (Wechselwirkung).
- **Auf einen Hintergrund begrenzt** (0 oder 1, dauerhaft aktiv). **Laufzeitverzögerung entfällt** (Annahme der momentanen Mischung; siehe ica-demo). Die **Quellenzahl wird als bekannt angenommen** (zu wenige Cluster verschmelzen Neuronen, zu viele teilen sie).
- **Synthetische Daten:** feste Spitzenform je Neuron, exakt lineare Mischung, weißes Gauß'sches Rauschen, Elektroden auf einer Zeile; Neuronenspalten sind positiv (alle Achsen im ersten Orthanten), deshalb ist das Zufallsniveau des Winkelfehlers vergleichsweise klein.
- **SCA nutzt die zeitliche Struktur nicht:** jeder Zeitpunkt wird unabhängig zerlegt.

## Verifikation

- Achsen-k-Means (drei Achsen im Vorzeichen-Zufall werden bis 0.5° erholt, Vorzeichen-Invarianz, Determinismus, einzelnes Cluster = Hauptachse); aktive Zeitpunkte und Kernpunkte (Teilmenge, rauschfrei keine Rest-Meldungen); Richtungen = Einheitsvektoren; **Kernpunkte gegen alle aktiven Zeitpunkte** (6 von 24 gegen 0 Fehlschläge).
- **L1-Rekonstruktion gegen `scipy.optimize.linprog`** (1-sparse exakt, 2-sparse gegen die LP-Lösung, Nebenbedingung erfüllt, L1-Norm nahe am Optimum); Einzelquelle exakt für 1-sparse; Ridge schrumpft; rauschfreie unterbestimmte Mischung (2 Elektroden, 4 Neuronen) wird mit Korrelation > 0.9 und Winkelfehler < 3° gelöst; eine Elektrode stürzt nicht ab.
- Szenario bit-genau gegen eingefrorene Werte aus ica-demo; ICA- und SOBI-Kopien gegen eingefrorene Werte (0.981 / 0.978, Toleranz 2e-3); Feuerraten-Faktor und Neuronen-Abstand wirken wie beschrieben; Winkelfehler und Zufallsniveau mit Handinstanzen.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Elektroden, Feuerrate und Überlappung, Neuronen-Abstand, Hintergrund, Rauschen, Länge, Preset-Hilfen; jeweils Mittel über die festen Sweep-Datensätze mit weiten Toleranzen); Verdict-Codes über sechs Datensätze;
  alle 6 Presets in weiten Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt auch mit einer Elektrode, Randgrößen, ausgeblendete Hintergrund-Art behält ihren Wert, Fenster-Klemmung, Sweeps und Szenen-Experiment), Achsensperre und explizite Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🧩 Szenen, 🚧 Grenzen, Mathe |
| `sca_algorithm.py` | aktive Zeitpunkte, Achsen-k-Means, L1/IRLS, Einzelquelle, `fit_sca` |
| `sca_ica.py`, `sca_sobi.py` | Vergleichsverfahren (wortgleich aus ica-demo/sobi-demo) |
| `sca_scenario.py`, `sca_constants.py` | Mehrelektroden-Generator mit Feuerrate, Neuronen-Abstand, Hintergrund; Konstanten, Presets |
| `sca_evaluation.py` | Zuordnung, Kennzahlen, Winkelfehler und Zufallsniveau, Sweeps, Szenen, Urteil |
| `sca_presets.py`, `sca_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Algorithmus, Kreuzprüfung (linprog), Szenario, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
