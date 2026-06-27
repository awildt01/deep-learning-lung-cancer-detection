# Deep Learning Lung Cancer Detection
**Work in Progress** 

<p align="left">
<img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/cplusplus-colored.svg" width="36"/>
<img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/python-colored.svg" width="36"/>
<img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/pytorch-colored.svg" width="36"/>
<img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/aws-colored.svg" width="36"/>
<img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/googlecloud-colored.svg" width="36"/>
</p>


Binäre Klassifizierungspipeline (Knoten vs. Nicht-Knoten) bei Computertomographien unter Verwendung des LUNA16-Datensatzes [LUNA16](https://luna16.grand-challenge.org/)  und PyTorch.

![Banner](docs/fixed_cnn_lung_tumor_detection.png)

[![Open in Hugging Face Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-ff9900?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/wildt/lung-cancer-detection)
&nbsp;
[![Gradio](https://img.shields.io/badge/Gradio-App-orange?style=for-the-badge)](https://huggingface.co/spaces/wildt/lung-cancer-detection)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

### Ergebnisse auf einen Blick

| Metrik | Wert |
| :--- | :---: |
| **ROC AUC** | 0.987 |
| **Average Precision (AP)** | 0.669 |
| **Recall @ Threshold 0.5** | 93,4 % (127/136 Knoten erkannt) |
| **Bester F1-Score** | 0.313 (Epoche 13) |
| **Trainingsepochen** | 20 (Google Colab GPU) |
| **Datensatz** | LUNA16 — ~551.000 Kandidaten |

<br>

## Inhaltsverzeichnis

- [Über das Projekt](#Über-das-Projekt)
- [Computertomographie](#Computertomographie)
  - [Was ist eine 3D-Tomographie?](#was-ist-eine-3d-tomographie)
  - [Räumliche Metadaten: Spacing und Origin](#räumliche-metadaten-spacing-und-origin)
  - [Anzeigen von Schnitten des Volumens](#anzeigen-von-schnitten-des-volumens)
  - [Fensterung (Windowing)](#fensterung-windowing)
- [Datenpipeline](#Datenpipeline)
- [Modell-Architektur](#Modell-Architektur)
- [Modellbewertung — Notebook 08](#modellbewertung--notebook-08)
  - [Trainingskurven](#trainingskurven)
  - [Zwei Checkpoints — eine klare Strategie](#zwei-checkpoints--eine-klare-strategie)
  - [Der Klassenungleichgewicht-Effekt](#der-klassenungleichgewicht-effekt)
  - [Konfusionsmatrix](#konfusionsmatrix)
  - [ROC-Kurve (AUC = 0.987)](#roc-kurve-auc--0987)
  - [Precision-Recall-Kurve (AP = 0.669)](#precision-recall-kurve-ap--0669)
  - [Threshold-Optimierung für den medizinischen Einsatz](#threshold-optimierung-für-den-medizinischen-einsatz)
  - [Wahrscheinlichkeitsverteilung](#wahrscheinlichkeitsverteilung)
  - [Fehleranalyse: Falsch-Negative und Falsch-Positive](#fehleranalyse-falsch-negative-und-falsch-positive)
  - [Exportiertes Inferenzmodul](#exportiertes-inferenzmodul)
- [Schnellstart — Inferenz](#schnellstart--inferenz)
- [Fortschritt](#Fortschritt)
- [Projektstruktur](#Projektstruktur)
- [Installation und Konfiguration](#Installation-und-Konfiguration)
- [Autor](#Autor)
- [Lizenz](#Lizenz)

<br>

## Über das Projekt

Das Projekt implementiert eine vollständige Pipeline zur Erkennung von Lungenknoten anhand von Computertomografien (CT-Scans), von der Datenerfassung und -aufbereitung bis hin zur Bereitstellung einer interaktiven Anwendung mit Gradio.

<p align="center">
  <img src="docs/fixed_landing_7_technical_flow_light.png" alt="Übersicht über die Pipeline" width="85%">
</p>
<p align="center"><em>Übersicht über die Pipeline – vom rohen CT-Scan bis zur Klassifizierung durch ein 3D-CNN.</em></p>

Der Ansatz nutzt vorberechnete Kandidaten, die vom LUNA16-Wettbewerb bereitgestellt werden (~551.000 XYZ-Koordinaten). Jeder Kandidat wird als 3D-Ausschnitt mit 32x48x48 Voxeln extrahiert und von einem 3D-CNN als Knoten oder Nicht-Knoten klassifiziert. Wir führen in der Hauptpipeline weder Segmentierung noch Erkennung durch – die Kandidaten werden bereits im Rahmen des LUNA16-Wettbewerbs vorberechnet.

<br>

## Computertomographie

<p align="center">
  <img src="docs/fixed_ct_slices_concept.png" alt="Slices einer Computertomographie" width="85%">
</p>
<p align="center"><em>Ein CT-Scan besteht aus Hunderten von übereinandergestapelten axialen Schichten, die ein 3D-Volumen bilden.</em></p>

Ein Computertomographie-Scan (CT-Scan) erzeugt ein 3D-Volumen des Körpers des Patienten. Jede „Scheibe“ (Slice) ist ein 2D-Bild, und der Stapel aus Scheiben bildet das gesamte Volumen. Die Werte jedes Voxels werden in Hounsfield-Einheiten (HU) gemessen – einer Skala, auf der Luft -1000 HU, Wasser 0 HU und Knochen bis zu +1000 HU beträgt.

Im Datensatz LUNA16 wird jeder CT-Scan als Paar aus einer .mhd-Datei (Metadaten) und einer .raw-Datei (Voxel) gespeichert. Die Aufgabe stellt zwei CSV-Dateien bereit: candidates.csv mit ~551.000 XYZ-Koordinaten verdächtiger Punkte und annotations.csv mit den von Radiologen bestätigten Knoten.

<br>

## Datenpipeline

<p align="center">
  <img src="docs/fixed_lung_cancer_pipeline_oreilly.png" alt="Datenpipeline" width="85%">
</p>
<p align="center"><em>Vollständige Pipeline: von den Rohdateien bis zum für das neuronale Netz bereitgestellten Sample.</em></p>

Der Weg der Rohdaten bis zum Eingang des neuronalen Netzes umfasst folgende Schritte:

1. **CT-Scan laden** — Einlesen der `.mhd-Datei` mit SimpleITK, um das 3D-Array und die Metadaten (Origin, Spacing, Direction) zu erhalten.
2. **Koordinaten konvertieren** — Die XYZ-Koordinaten (Millimeter des Patienten) werden in IRC-Indizes (Index, Row, Col) des NumPy-Arrays konvertiert.
3. **3D-Ausschnitt extrahieren** — ein Ausschnitt (Patch) von 32x48x48 Voxeln wird um jeden Kandidaten herum ausgeschnitten.
4. **PyTorch-Sample erstellen** — der Ausschnitt wird zu einem Tensor `[1, 32, 48, 48]`, bereit für den DataLoader.

<br>

Ein CT-Scan ist im Wesentlichen ein dreidimensionales Array, in dem jedes Voxel die Röntgenstrahlabschwächung in Hounsfield-Einheiten quantifiziert. Diese standardisierte numerische Darstellung ermöglicht es, anatomische Strukturen zu segmentieren, neuronale Netze zur Erkennung von Läsionen zu trainieren und reproduzierbare Analyse-Pipelines zu erstellen – alles auf der Grundlage von Operationen mit NumPy-Arrays. 

## Was ist eine 3D-Tomographie?

Eine Computertomographie (CT-Scan) ist eine Untersuchung, die ein 3D-Volumen des Körperinneren erzeugt. Das Gerät sendet Röntgenstrahlen aus verschiedenen Winkeln aus und rekonstruiert eine Reihe von Querschnitten des Patienten.

In der Praxis ist jeder CT-Scan, den Sie in Python bearbeiten werden, ein dreidimensionales NumPy-Array. Jede Position in diesem Array ist ein Voxel (das 3D-Äquivalent eines Pixels), und der gespeicherte numerische Wert wird in Hounsfield-Einheiten (HU) gemessen, einer standardisierten physikalischen Skala:

- Luft: −1000 HU
- Fett: −120 bis −60 HU
- Wasser: 0 HU
- Weichgewebe (Muskeln, Organe): +40 bis +80 HU
- Aufgeblähte Lunge: −950 bis −500 HU
- Kompakter Knochen: +1000 HU oder mehr

Im Gegensatz zu gewöhnlichen Bildern (bei denen Pixelwerte willkürlich sind) hat bei einem CT-Scan jede Zahl eine physikalische Bedeutung. Diese Eigenschaft ermöglicht es, anatomische Strukturen durch einfache Schwellenwertoperationen herauszufiltern.


<p align="center">
  <img src="docs/01_Pixelwerte.png" alt="Pixelwerte" width="85%">
</p>
<p align="center"><em>Pixelwerte in Hounsfield-Einheiten (HU)</em></p>

<br>

## Räumliche Metadaten: Spacing und Origin

Neben den Voxeln enthält ein CT-Scan räumliche Metadaten, die das Array mit der physikalischen Welt verknüpfen:

```python
spacing = ct_sitk.GetSpacing()
origin = ct_sitk.GetOrigin()

print(f"Spacing (x, y, z): {spacing} mm")
print(f"Origin: {origin}")
```

Ausgabe:
```text
spacing (x, y, z): (0.742, 0.742, 2.5) mm
Origin: (-182.5, -190.0, -313.75)
```


Der Abstand (spacing) gibt den physikalischen Abstand in Millimetern zwischen aufeinanderfolgenden Voxeln an. In diesem Fall misst jedes Pixel innerhalb einer Schicht 0,742 mm, der Abstand zwischen den Schichten beträgt jedoch 2,5 mm. Das bedeutet, dass das Volumen anisotrop ist: Die Auflösung ist nicht in allen Richtungen gleichmäßig.

Der Ursprung (origin) gibt die Position des ersten Voxels im Koordinatensystem des Patienten an. Er ist unerlässlich für die Umrechnung zwischen Array-Indizes und tatsächlichen Koordinaten in Millimetern.

<br>

## Anzeigen von Schnitten des Volumens

Die einfachste Möglichkeit, einen CT-Scan anzuzeigen, besteht darin, einzelne Schnitte mit Matplotlib darzustellen. Da jeder Schnitt ein 2D-Array ist, genügt es, die erste Achse zu indizieren:

```python
# Einzelnen Schnitt anzeigen
schicht = ct_array[ct_array.shape[0] // 2]
plt.imshow(schicht, cmap="gray")
plt.axis("off")
plt.show()
```

<p align="center">
  <img src="docs/02_Volume.png" alt="Schnitt des Volumens" width="50%">
</p>
<p align="center"><em>Visualisierung einer einzelnen Schicht</em></p>

Alternativ können mehrere über das Volumen verteilte Schnitte gleichzeitig visualisiert werden, um einen schnellen Überblick zu erhalten:

```python
# 5 Schichten über das Volumen verteilt anzeigen
n_schichten = ct_array.shape[0]
indices = [0, n_schichten // 4, n_schichten // 2, 3 * n_schichten // 4, n_schichten - 1]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, idx in zip(axes, indices):
    ax.imshow(ct_array[idx], cmap="gray")
    ax.set_title(f"Schicht {idx}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

<p align="center">
  <img src="docs/03_Schichten.png" alt="Fünf verteilte Schichten" width="85%">
</p>
<p align="center"><em>Visualisierung mehrerer Schichten über das gesamte Volumen hinweg</em></p>

<br>

## Fensterung (Windowing): Kontrastoptimierung für medizinische Bilddaten

Ein entscheidender Unterschied zwischen der reinen Anzeige eines Bildes und dem tiefen Verständnis medizinischer Bilddaten liegt in der **Fensterung (Windowing)**.

### Das biologische und physikalische Problem
Ein moderner CT-Scan besitzt einen enormen Dynamikbereich (meist 12-Bit-Darstellung). Im LUNA16-Datensatz reichen die Voxelwerte von **$-2048$ bis $+3071$ Hounsfield-Einheiten (HU)**. Der exakte Bereich hängt von den DICOM-Metadaten (`Rescale Intercept` und `Rescale Slope`) ab.

Daraus ergibt sich ein doppeltes Problem:
1. **Das physikalische Datenvolumen:** Ein CT-Scan deckt eine Spanne von über 4.000 HU ab.
2. **Die biologische Grenze:** Das menschliche Auge kann auf einem Monitor nur etwa **60 bis 80 Graustufen** gleichzeitig differenzieren.

Würde man den gesamten Wertebereich linear auf eine Grauskala abbilden, wären Gewebestrukturen mit ähnlicher Dichte (z. B. gesundes Hirngewebe und ein Blutgerinnsel, die sich nur um wenige HU unterscheiden) ununterscheidbar. Das Bild wäre ein kontrastarmer, grauer Einheitsbrei.

### Die Lösung: Kontrastspreizung durch Level (L) und Width (W)
Die Fensterung fungiert wie eine **Lupe für Kontraste**. Sie wählt einen relevanten HU-Teilbereich aus und streckt diesen linear auf die volle Graustufenskala des Monitors. Gesteuert wird dies über zwei Parameter:
* **Window Level (Zentrum / $L$):** Bestimmt den Mittelpunkt des HU-Bereichs, den man untersuchen möchte. Dieser HU-Wert wird als mittleres Grau dargestellt.
* **Window Width (Breite / $W$):** Bestimmt die Breite des HU-Fensters. Ein schmales Fenster erhöht den Kontrast im Zielbereich drastisch; ein breites Fenster zeigt mehr unterschiedliche Gewebearten gleichzeitig, jedoch mit weniger Kontrast.

Die mathematische Abbildung auf dem Monitor berechnet sich wie folgt:

$$\text{Untergrenze} = L - \frac{W}{2} \quad \implies \text{Alles unter dieser Grenze wird rein schwarz dargestellt.}$$
$$\text{Obergrenze} = L + \frac{W}{2} \quad \implies \text{Alles über dieser Grenze wird rein weiß dargestellt.}$$

Der Wertebereich dazwischen wird linear auf die Graustufen des Monitors $[0, 255]$ skaliert.

### Standardfenster in der klinischen Praxis
Je nach diagnostischer Fragestellung schalten Radiologen zwischen verschiedenen Fenstereinstellungen um:

| Fenstername | Zentrum ($L$) | Breite ($W$) | Hauptziel |
| :--- | :---: | :---: | :--- |
| **Lungenfenster** | $-600\text{ HU}$ | $1600\text{ HU}$ | Feinste Strukturen der Lungenbläschen (Alveolen) und Lungengefäße. |
| **Weichteilfenster** (Mediastinum) | $+50\text{ HU}$ | $350\text{ HU}$ | Abgrenzung von Organen und Weichteilgewebe (Herz, Leber, Gefäße). |
| **Knochenfenster** | $+300\text{ HU}$ | $2000\text{ HU}$ | Visualisierung von Frakturen und der inneren Knochenstruktur (Spongiosa). |
| **Hirnfenster** | $+35\text{ HU}$ | $80\text{ HU}$ | Extrem schmal, um feinste Dichteunterschiede (z.B. frische Blutungen) im Gehirn zu erkennen. |

### Klinisches Beispiel: Ein Brustkorb-Schnitt (Thorax), zwei Welten
Betrachten wir eine CT-Schicht des Brustkorbs mit Herz, Lunge, Muskeln und Rippen:
1. **Im Weichteilfenster ($L$: $+50$, $W$: $350$):**
   * Der sichtbare Bereich liegt bei $[-125\text{ HU}, +225\text{ HU}]$.
   * Organe und Muskeln liegen im Zentrum und zeigen feine Kontraste (z. B. Herzgewebe vs. Blutgefäße).
   * Knochen ($> +400\text{ HU}$) überschreiten die Obergrenze und erscheinen als strukturloses, grelles Weiß.
   * Das Lungengewebe ($\approx -700\text{ HU}$) unterschreitet die Untergrenze und erscheint komplett schwarz.
2. **Im Knochenfenster ($L$: $+300$, $W$: $2000$):**
   * Der sichtbare Bereich liegt bei $[-700\text{ HU}, +1300\text{ HU}]$.
   * Knochen liegen perfekt im Sichtfeld. Details wie die harte Außenschale (Kompakta) und das schwammartige Innenleben (Spongiosa) werden sichtbar.
   * Weichgewebe verschwimmt am unteren Ende der Skala zu einem einheitlich dunklen Graubrei.

### Implementierung in Python
Der folgende Python-Code zeigt, wie die Fensterung mit NumPy implementiert und Matplotlib zur Visualisierung genutzt wird. Während Matplotlib Werte bei `imshow` automatisch auf die Grauskala normiert, ist in einer Deep-Learning-Produktionspipeline eine explizite Normalisierung auf $[0, 1]$ oder $[0, 255]$ vor der Modelleingabe unerlässlich.

```python
import numpy as np
import matplotlib.pyplot as plt

def apply_window(img, center, width):
    """Wendet eine Fensterung (Windowing) auf ein CT-Scan-Bild an."""
    lower = center - width // 2
    upper = center + width // 2
    return np.clip(img, lower, upper)

# Beispiel zur Visualisierung des Effekts auf derselben Schicht
half_slice = ct_array[ct_array.shape[0] // 2]

slice_views = {
    "Lungenfenster (C:-600, W:1600)": (-600, 1600),
    "Mediastinalfenster (C:50, W:350)": (50, 350),
    "Knochenfenster (C:300, W:2000)": (300, 2000),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (name, (center, width)) in zip(axes, slice_views.items()):
    img = apply_window(half_slice, center, width)
    ax.imshow(img, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

<p align="center">
  <img src="docs/04_fenster.png" alt="Windowing-Effekt auf CT-Schnitt" width="85%">
</p>
<p align="center"><em>Derselbe CT-Schnitt unter Anwendung dreier unterschiedlicher Fensterungen.</em></p>

<br>

---

## Modell-Architektur

Das Netzwerk wurde in **PyTorch** implementiert und besteht aus zwei Kernkomponenten: einem wiederverwendbaren `LunaBlock` und dem übergeordneten `LunaModel`.

### LunaBlock — der Baustein

Jeder Block enthält:
* **2 × Conv3d** (Kernel 3×3×3, Padding 1) — Extraktion räumlicher Features über alle drei Dimensionen hinweg.
* **ReLU-Aktivierung** nach jeder Faltung.
* **3D Max-Pooling** (2×2×2) — Downsampling der Feature-Maps zur Reduktion der Dimensionalität.

### LunaModel — die Gesamtarchitektur

```
Eingabe: [B, 1, 32, 48, 48]   ← 3D-Ausschnitt (1 Kanal, Grauwert)

  ┌─ BatchNorm3d(1)            ← Normalisierung der Rohdaten
  │
  ├─ LunaBlock 1:   1  →  8 Kanäle   (Ausgabe: 16×24×24)
  ├─ LunaBlock 2:   8  → 16 Kanäle   (Ausgabe: 8×12×12)
  ├─ LunaBlock 3:  16  → 32 Kanäle   (Ausgabe: 4×6×6)
  ├─ LunaBlock 4:  32  → 64 Kanäle   (Ausgabe: 2×3×3)
  │
  ├─ Flatten                   ← 64 × 2 × 3 × 3 = 1.152 Features
  ├─ Linear(1152, 2)           ← Fully-Connected-Layer (2 Klassen)
  └─ Softmax(dim=1)            ← Wahrscheinlichkeiten: [P(kein Knoten), P(Knoten)]
```

### Zusammenfassung

| Eigenschaft | Details |
| :--- | :--- |
| **Framework** | PyTorch (≥ 2.2) |
| **Typ** | 3D-CNN (binäre Klassifikation) |
| **Eingabe** | `[B, 1, 32, 48, 48]` — einzelner Grauwert-Kanal |
| **Blöcke** | 4 × LunaBlock (je 2 Conv3d + ReLU + MaxPool3d) |
| **Kanalprogression** | 1 → 8 → 16 → 32 → 64 |
| **Klassifikationskopf** | Linear(1152, 2) + Softmax |
| **Normalisierung** | BatchNorm3d am Eingang |
| **Gewichtsinitialisierung** | Kaiming Normal (fan-out, ReLU) |
| **Ausgabe** | Logits + Wahrscheinlichkeiten (2 Klassen) |

<br>

---

## Modellbewertung — Notebook 08

Nach 20 Trainingsepochen auf Google Colab (GPU) wird das trainierte `LunaModel` in diesem Notebook einer systematischen, medizinisch orientierten Auswertung unterzogen. Das Ziel: nicht nur eine einzige Kennzahl ablesen, sondern das Verhalten des Modells wirklich **verstehen** — inklusive seiner Fehler.

<br>

### Trainingskurven

Bevor wir das Modell auf dem Validierungsset auswerten, lohnt sich ein Blick auf den Trainingsverlauf über alle 20 Epochen.

<p align="center">
  <img src="docs/evaluation_training_curves.png" alt="Trainingskurven: Loss und Metriken über 20 Epochen" width="95%">
</p>
<p align="center"><em>Links: Trainings- und Validierungs-Loss über 20 Epochen — beide sinken gemeinsam ohne große Divergenz (kein Overfitting). Rechts: Recall, Präzision und F1-Score auf dem Validierungsset. Der beste Checkpoint (Epoche 13, F1 = 0.313) wird für die finale Auswertung geladen.</em></p>

<br>

### Zwei Checkpoints — eine klare Strategie

Während des Trainings werden zwei Checkpoints gespeichert:

| Checkpoint | Beschreibung |
| :--- | :--- |
| `luna_model_best.pt` | Epoche mit dem besten Validierungs-F1-Score (Epoche 13, F1 = 0.313) |
| `luna_model_latest.pt` | Letzter Trainingsstand (Epoche 20) mit vollständiger Verlaufshistorie |

Die **Gewichte** stammen aus dem Best-Checkpoint (Epoche 13), die **Trainingshistorie** (alle 20 Epochen) aus dem Latest-Checkpoint. Dadurch ergibt sich das vollständige Bild: Wie hat sich das Modell entwickelt, und welcher Stand ist für den Einsatz geeignet?

> **Warum der F1-Score als Kriterium?** Bei einem stark unbalancierten Datensatz (~0,25 % echte Knoten) ist der F1-Score robuster als bloße Genauigkeit (Accuracy), da er sowohl Präzision als auch Recall berücksichtigt.

<br>

### Der Klassenungleichgewicht-Effekt

Ein wichtiges Detail des Trainings: Die **Trainingsmetriken** sind bewusst überhöht, weil balancierte Batches verwendet wurden (50 % Knoten, 50 % Nicht-Knoten). Auf dem Validierungsset hingegen spiegelt sich die **reale Verteilung** wider (~0,25 % Knoten). Folglich fallen Präzision und F1-Score auf dem Validierungsset deutlich niedriger aus — dies ist kein Anzeichen für ein schlechtes Modell, sondern ein erwartetes und bekanntes Phänomen.

<br>

### Konfusionsmatrix

Die Konfusionsmatrix bei Threshold 0,5 zeigt das Ergebnis auf dem gesamten Validierungsset auf einen Blick:

<p align="center">
  <img src="docs/nb08_plot_020_001.png" alt="Konfusionsmatrix bei Threshold 0.5" width="55%">
</p>
<p align="center"><em>Konfusionsmatrix (Threshold = 0.5): Das Modell erkennt 127 von 136 echten Knoten korrekt (Recall ≈ 93 %) und produziert 1.476 Fehlalarme bei 54.971 Nicht-Knoten.</em></p>

Die Zahlen im Überblick:

| | Vorhergesagt: Kein Knoten | Vorhergesagt: Knoten |
| :--- | :---: | :---: |
| **Tatsächlich: Kein Knoten** | 53.495 ✅ (TN) | 1.476 ❌ (FP) |
| **Tatsächlich: Knoten** | 9 ❌ (FN) | 127 ✅ (TP) |

<br>

### ROC-Kurve (AUC = 0.987)

Die ROC-Kurve zeigt den Trade-off zwischen Recall (Sensitivität) und Falsch-Positiv-Rate über alle möglichen Schwellenwerte. Ein AUC-Wert nahe 1,0 bedeutet eine exzellente Unterscheidungsfähigkeit.

<p align="center">
  <img src="docs/nb08_plot_023_000.png" alt="ROC-Kurve AUC=0.987" width="55%">
</p>
<p align="center"><em>ROC-Kurve: AUC = 0.987 — das Modell trennt Knoten von Nicht-Knoten deutlich besser als ein zufälliger Klassifikator (gestrichelte Linie).</em></p>

<br>

### Precision-Recall-Kurve (AP = 0.669)

Die ROC-Kurve kann bei stark unbalancierten Datensätzen zu optimistisch wirken. Die Precision-Recall-Kurve (AP = Average Precision) ist hier aussagekräftiger — sie zeigt direkt, wie gut das Modell echte Knoten findet, ohne Fehlalarme zu erzeugen.

<p align="center">
  <img src="docs/nb08_plot_026_000.png" alt="Precision-Recall-Kurve AP=0.669" width="55%">
</p>
<p align="center"><em>Precision-Recall-Kurve: AP = 0.669. Bei hohem Recall (> 80 %) sinkt die Präzision deutlich — ein klassisches Merkmal des Klassenungleichgewichts in medizinischen Datensätzen.</em></p>

<br>

### Threshold-Optimierung für den medizinischen Einsatz

Der Standard-Schwellenwert von 0,5 ist für medizinische Anwendungen oft nicht optimal. Das Notebook analysiert systematisch alle Schwellenwerte und deren Auswirkung auf den Trade-off zwischen Sensitivität und Spezifität.

<p align="center">
  <img src="docs/nb08_plot_030_000.png" alt="Metriken vs. Threshold" width="65%">
</p>
<p align="center"><em>Recall, Präzision und F1-Score als Funktion des Schwellenwerts. Ein niedriger Threshold hält den Recall hoch (fast alle Knoten werden erkannt), erhöht aber die Fehlalarmrate.</em></p>

> **Medizinische Implikation:** In der Krebsfrüherkennung ist ein **hoher Recall** (wenige übersehene Knoten) wichtiger als hohe Präzision. Ein angepasster Schwellenwert kann die Sensitivität deutlich verbessern — auf Kosten von mehr Fehlalarmen, die durch Radiologen überprüft werden können.

<br>

### Wahrscheinlichkeitsverteilung

Ein weiterer Blick auf das Modellverhalten: die Verteilung der vorhergesagten Wahrscheinlichkeiten getrennt nach Klasse. Eine gute Trennung der Verteilungen deutet auf eine hohe Modellsicherheit hin.

<p align="center">
  <img src="docs/nb08_plot_045_000.png" alt="Verteilung der Wahrscheinlichkeiten nach Klasse" width="65%">
</p>
<p align="center"><em>Wahrscheinlichkeitsverteilung nach Klasse: Nicht-Knoten häufen sich nahe 0 (das Modell ist sicher, dass es kein Knoten ist), echte Knoten häufen sich nahe 1 (das Modell erkennt sie mit hoher Konfidenz).</em></p>

<br>

### Fehleranalyse: Falsch-Negative und Falsch-Positive

Das Notebook visualisiert die schwerwiegendsten Fehler des Modells direkt als CT-Ausschnitte:

**Falsch-Negative (FN) — übersehene Knoten ⚠️**

Echte Knoten, die das Modell mit Threshold 0,5 nicht erkannt hat — sortiert nach aufsteigender Konfidenz (die am wenigsten sicheren Fehler zuerst):

<p align="center">
  <img src="docs/nb08_plot_036_000.png" alt="Falsch-Negative: übersehene Knoten" width="95%">
</p>
<p align="center"><em>Falsch-Negative Kandidaten: echte Lungenknoten, die das Modell übersehen hat. Die Konfidenz reicht von 0.017 bis 0.236 — einige dieser Fälle sind auch für das menschliche Auge schwer zu erkennen.</em></p>

**Falsch-Positive (FP) — Fehlalarme**

Kandidaten, die das Modell fälschlicherweise als Knoten markiert hat — sortiert nach absteigender Konfidenz (die „überzeugendsten" Fehlalarme):

<p align="center">
  <img src="docs/nb08_plot_038_000.png" alt="Falsch-Positive: Fehlalarme" width="95%">
</p>
<p align="center"><em>Falsch-Positive Kandidaten: Das Modell klassifiziert diese Kandidaten mit Konfidenz 1.000 als Knoten — sie sind es jedoch nicht. Strukturen wie Blutgefäße oder Gewebeübergänge können morphologisch knotenartig erscheinen.</em></p>

**Drei anatomische Ebenen im Vergleich**

Für den kritischsten Falsch-Negativ-Fall wird der CT-Ausschnitt in allen drei Standardebenen dargestellt:

<p align="center">
  <img src="docs/nb08_plot_041_000.png" alt="Falsch-Negativer Knoten in drei anatomischen Ebenen" width="75%">
</p>
<p align="center"><em>Ein übersehener Knoten (prob = 0.017) in axialer, koronaler und sagittaler Ansicht. Die drei Ebenen zeigen, warum dieser Fall schwierig ist: Der Kandidat liegt in einer Region mit komplexen anatomischen Strukturen.</em></p>

<br>

### Exportiertes Inferenzmodul

Als letzter Schritt exportiert das Notebook ein produktionsreifes Modul `src/inference.py` mit allen Funktionen, die für den Einsatz mit dem Gradio-Interface (Notebook 09) benötigt werden. Dieses Modul kapselt die gesamte Inferenzlogik — vom Laden des Modells bis zur Rückgabe der Klassifizierungsergebnisse.

<br>

---

## Schnellstart — Inferenz

So klassifizieren Sie die Kandidaten eines einzelnen CT-Scans mit einem trainierten Checkpoint:

```python
import torch
from src.inference import load_model, classify_ct

# 1. Modell laden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, info = load_model("checkpoints/luna_model_best.pt", device=device)
print(f"Checkpoint: Epoche {info['epoch']}, F1 = {info['best_f1']:.3f}")

# 2. Alle Kandidaten eines CT-Scans klassifizieren
series_uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860"
results = classify_ct(series_uid, model, device)

# 3. Top-5 Kandidaten mit höchster Knoten-Wahrscheinlichkeit
for r in results[:5]:
    print(f"  P(Knoten) = {r['probability']:.3f}  |  Koordinaten: {r['center_xyz']}")
```

> **Voraussetzung:** Der LUNA16-Datensatz muss unter `data/` verfügbar sein (siehe [Installation und Konfiguration](#Installation-und-Konfiguration)).

<br>

---


## Fortschritt

- [x] Herunterladen und Aufbereiten des LUNA16-Datensatzes
- [x] Explorative Analyse und Zusammenführung der Datenquellen
- [x] Einlesen der CT-Scans und Koordinatenumwandlung
- [x] Erstellung des PyTorch-Datensatzes mit Extraktion von 3D-Ausschnitten
- [x] Architektur des 3D-CNN zur Klassifizierung von Knoten
- [x] Trainingsschleife mit Datenausgleich und Datenvergrößerung
- [x] Vollständiges Training auf der GPU
- [x] Modellbewertung und Fehleranalyse
- [x] Bereitstellung mit Gradio

<br>



## Projektstruktur

```
├── notebooks/                 Jupyter-Notebooks 
├── src/                       Python-Module
│   ├── data/
│   ├── models/
│   └── visualization/
├── tests/                   Automatisierte Tests
├── docs/                    Diagramme und Referenzen
└── pyproject.toml           Abhängigkeiten und Konfiguration
```

<br>

## Installation und Konfiguration

### Systemvoraussetzungen

| Anforderung | Minimum | Empfohlen |
| :--- | :--- | :--- |
| **Python** | ≥ 3.11.3 | 3.11.x oder 3.12.x |
| **GPU** | Nicht erforderlich (CPU-Inferenz möglich) | NVIDIA-GPU mit CUDA-Unterstützung |
| **CUDA** | — | CUDA 11.8 oder 12.x (passend zur PyTorch-Version) |
| **RAM** | 8 GB | ≥ 16 GB (CT-Scans sind speicherintensiv) |
| **Festplatte** | ~2 GB (Code + Modell) | ~250 GB (inkl. LUNA16-Datensatz) |
| **Paketverwaltung** | [UV](https://docs.astral.sh/uv/) | — |

> **Hinweis zur GPU:** Das Training des 3D-CNN wurde auf Google Colab (NVIDIA T4/A100) durchgeführt. Für reine Inferenz reicht eine CPU aus, das Training ohne GPU ist jedoch nicht praktikabel. PyTorch wird in der Version `≥ 2.2, < 2.6` verwendet — stellen Sie sicher, dass Ihre CUDA-Toolkit-Version dazu kompatibel ist ([PyTorch-Kompatibilitätsmatrix](https://pytorch.org/get-started/locally/)).

### Schritte

1. Klonen Sie das Repository auf Ihren lokalen Rechner:

```bash
git clone https://github.com/awildt01/deep-learning-lung-cancer-detection.git
cd deep-learning-lung-cancer-detection
```

2. Installieren Sie die Abhängigkeiten mit UV:

```bash
uv sync
```

3. Aktivieren Sie die virtuelle Umgebung:

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

<br>

---

## Autor

**Carlos Melo** ([@awildt01](https://github.com/awildt01))

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).
