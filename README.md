
# Deep Learning Lung Cancer Detection

<a href="https://docs.microsoft.com/en-us/cpp/?view=msvc-170" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/cplusplus-colored.svg" width="36" height="36" alt="C++" /></a><a href="https://www.python.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/python-colored.svg" width="36" height="36" alt="Python" /></a><a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/tensorflow-colored.svg" width="36" height="36" alt="TensorFlow" /></a><a href="https://pytorch.org/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/pytorch-colored.svg" width="36" height="36" alt="PyTorch" /></a><a href="https://aws.amazon.com" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/aws-colored.svg" width="36" height="36" alt="Amazon Web Services" /></a><a href="https://cloud.google.com/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/googlecloud-colored.svg" width="36" height="36" alt="Google Cloud" /></a> </p>
<br>





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
│   ├── 01_download_luna16
│   ├── 02_explore_csv_data
│   ├── 03_analyze_coordinates
│   ├── 04_ct_scan_to_dataset
│   ├── 05_model_architecture
│   ├── 06_training
│   ├── 07_colab_training
│   ├── 08_model_evaluation
│   └── 09_gradio_deploy
├── src/                       Python-Module (generiert über %%writefile)
│   ├── luna_data.py
│   ├── model.py
│   ├── training.py
│   └── inference.py
├── app.py                   Gradio-Anwendung (generiert über %%writefile)
├── tests/                   Automatisierte Tests
├── checkpoints/             Checkpoints des trainierten Modells
├── data/                    LUNA16-Datensatz (nicht versioniert)
├── docs/                    Diagramme und Referenzen
└── pyproject.toml           Abhängigkeiten und Konfiguration
```

<br>

## Installation und Konfiguration

1. Clonar o repositório para a sua máquina local:

```bash
git clone https://github.com/carlosfab/bootcamp-deep-learning.git
cd bootcamp-deep-learning
```

2. Instalar as dependências com [UV](https://docs.astral.sh/uv/):

```bash
uv sync
```

3. Ativar o ambiente virtual:

```bash
source .venv/bin/activate
```

4. Rodar os testes para verificar que está tudo funcionando:

```bash
pytest tests/ -v
```

5. O dataset LUNA16 (~111 GB) deve ser baixado separadamente. O notebook `01_download_luna16.ipynb` contém as instruções de download via API.

---

Projeto desenvolvido como parte do Bootcamp de Deep Learning para Visão Computacional da [STAR Research Institute](https://starresearch.institute).
