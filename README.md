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

<br>

## Inhaltsverzeichnis

- [Über das Projekt](#Über-das-Projekt)
- [Computertomographie](#Computertomographie)
  - [Was ist eine 3D-Tomographie?](#was-ist-eine-3d-tomographie)
  - [Räumliche Metadaten: Spacing und Origin](#räumliche-metadaten-spacing-und-origin)
- [Datenpipeline](#Datenpipeline)
- [Fortschritt](#Fortschritt)
- [Projektstruktur](#Projektstruktur)
- [Installation und Konfiguration](#Installation-und-Konfiguration)

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

Die einfachste Möglichkeit, einen CT-Scan anzuzeigen, besteht darin, einzelne Schnitte mit matplotlib darzustellen. Da jeder Schnitt ein 2D-Array ist, genügt es, die erste Achse zu indizieren:


```python
# erste Schnitt
Scheibe  = ct_array[0]
plt.imshow(Scheibe , cmap="gray")
plt.axis("off")
plt.show()
```



<p align="center">
  <img src="docs/02_Volume.png" alt="volum" width="50%">
</p>
<p align="center"><em>Pixelwerte in Hounsfield-Einheiten (HU)</em></p>


Anzeigen von Schnitten des Volumens
Die einfachste Möglichkeit, einen CT-Scan anzuzeigen, besteht darin, einzelne Schnitte mit matplotlib darzustellen. Da jeder Schnitt ein 2D-Array ist, genügt es, die erste Achse zu indizieren:

<p align="center">
  <img src="docs/03_Schichten.png" alt="schichten" width="85%">
</p>
<p align="center"><em>Pixelwerte in Hounsfield-Einheiten (HU)</em></p>



<br>

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
.venv\Scripts\activate
```
