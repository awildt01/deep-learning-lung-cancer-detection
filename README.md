# Deep Learning Lung Cancer Detection

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

## Sumário

- [Über das Projekt](#Über-das-Projekt)
- [Computertomographie](#Computertomographie)
- [Datenpipeline](#Datenpipeline)
- [Fortschritt](#Fortschritt)
- [Projektstruktur](#Projektstruktur)
- [Installation und Konfiguration](#Installation-und-Konfiguration)

<br>

## Über das Projekt

Das Projekt implementiert eine vollständige Pipeline zur Erkennung von Lungenknoten anhand von Computertomografien (CT-Scans), von der Datenerfassung und -aufbereitung bis hin zur Bereitstellung einer interaktiven Anwendung mit Gradio.

<p align="center">
  <img src="docs/fixed_landing_7_technical_flow_light.png" alt="Visão geral do pipeline" width="85%">
</p>
<p align="center"><em>Übersicht über die Pipeline – vom rohen CT-Scan bis zur Klassifizierung durch ein 3D-CNN.</em></p>

Der Ansatz nutzt vorberechnete Kandidaten, die vom LUNA16-Wettbewerb bereitgestellt werden (~551.000 XYZ-Koordinaten). Jeder Kandidat wird als 3D-Ausschnitt mit 32x48x48 Voxeln extrahiert und von einem 3D-CNN als Knoten oder Nicht-Knoten klassifiziert. Wir führen in der Hauptpipeline weder Segmentierung noch Erkennung durch – die Kandidaten werden bereits im Rahmen des LUNA16-Wettbewerbs vorberechnet.

<br>

## Computertomographie

<p align="center">
  <img src="docs/fixed_ct_slices_concept.png" alt="Slices de uma tomografia computadorizada" width="85%">
</p>
<p align="center"><em>Ein CT-Scan besteht aus Hunderten von übereinandergestapelten axialen Schichten, die ein 3D-Volumen bilden.</em></p>

Ein Computertomographie-Scan (CT-Scan) erzeugt ein 3D-Volumen des Körpers des Patienten. Jede „Scheibe“ (Slice) ist ein 2D-Bild, und der Stapel aus Scheiben bildet das gesamte Volumen. Die Werte jedes Voxels werden in Hounsfield-Einheiten (HU) gemessen – einer Skala, auf der Luft -1000 HU, Wasser 0 HU und Knochen bis zu +1000 HU beträgt.

Im Datensatz LUNA16 wird jeder CT-Scan als Paar aus einer .mhd-Datei (Metadaten) und einer .raw-Datei (Voxel) gespeichert. Die Aufgabe stellt zwei CSV-Dateien bereit: candidates.csv mit ~551.000 XYZ-Koordinaten verdächtiger Punkte und annotations.csv mit den von Radiologen bestätigten Knoten.

<br>

## Datenpipeline

<p align="center">
  <img src="docs/fixed_lung_cancer_pipeline_oreilly.png" alt="Pipeline de dados" width="85%">
</p>
<p align="center"><em>Vollständige Pipeline: von den Rohdateien bis zum für das neuronale Netz bereitgestellten Sample.</em></p>

Der Weg der Rohdaten bis zum Eingang des neuronalen Netzes umfasst folgende Schritte:

1. **CT-Scan laden** — Einlesen der `.mhd-Datei` mit SimpleITK, um das 3D-Array und die Metadaten (Origin, Spacing, Direction) zu erhalten.
2. **Koordinaten konvertieren** — Die XYZ-Koordinaten (Millimeter des Patienten) werden in IRC-Indizes (Index, Row, Col) des NumPy-Arrays konvertiert.
3. **3D-Ausschnitt extrahieren** — um patch de 32x48x48 voxels é recortado ao redor de cada candidato.
4. **PyTorch-Sample erstellen** — der Ausschnitt wird zu einem Tensor `[1, 32, 48, 48]`, bereit für den DataLoader.

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
