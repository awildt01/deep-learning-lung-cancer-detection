"""Gradio Demo – Lung Cancer Nodule Classifier (3D CNN, LUNA16)."""

import io
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import LunaModel

# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
# Absoluter Pfad zur .pt-Datei (robust gegen wechselndes Working Dir auf HF Spaces)
_HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(_HERE, "luna_model_best.pt")
DEVICE = torch.device("cpu")


def load_model():
    model = LunaModel()
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: '{CHECKPOINT}'\n"
            f"__file__={__file__}\n"
            f"cwd={os.getcwd()}\n"
            f"Dateien in _HERE: {os.listdir(_HERE)}"
        )
    size_mb = os.path.getsize(CHECKPOINT) / 1e6
    print(f"[INFO] Lade Checkpoint: {CHECKPOINT} ({size_mb:.2f} MB)")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    epoch = ckpt.get('epoch', '?')
    best_f1 = ckpt.get('best_f1', 0)
    print(f"[INFO] Checkpoint OK: epoch={epoch}, best_f1={best_f1:.4f}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("[INFO] Modell erfolgreich geladen und bereit.")
    return model


MODEL = load_model()

# ---------------------------------------------------------------------------
# Demo-Patches mit realistischen Hounsfield-Einheiten (HU)
# Lunge: ca. -700 HU | Knoten (Weichteil): ca. +30 HU
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

# Knoten-Patch: Lungengewebe als Hintergrund + kugelförmiger Weichteil-Knoten
DEMO_NODULE = rng.normal(loc=-700, scale=60, size=(32, 48, 48)).astype(np.float32)
sphere_center = (16, 24, 24)
for i in range(32):
    for j in range(48):
        for k in range(48):
            d = np.sqrt((i - sphere_center[0])**2 +
                        (j - sphere_center[1])**2 +
                        (k - sphere_center[2])**2)
            if d < 6:
                DEMO_NODULE[i, j, k] = rng.normal(30, 20)

# Nicht-Knoten-Patch: reines Lungengewebe, kein Weichteilelement
DEMO_NON_NODULE = rng.normal(loc=-720, scale=80, size=(32, 48, 48)).astype(np.float32)


def predict_patch(patch_array: np.ndarray):
    """Führt Inferenz auf einem 3D-Patch (32x48x48) durch."""
    patch_t = torch.from_numpy(patch_array).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _, probs = MODEL(patch_t)
    return float(probs[0, 1])


def buf_to_pil(buf: io.BytesIO) -> Image.Image:
    buf.seek(0)
    return Image.open(buf).copy()


def visualize_patch(patch_array: np.ndarray, prob: float, threshold: float) -> Image.Image:
    """Erstellt eine Visualisierung mit 3 Schichten und Wahrscheinlichkeitsbalken."""
    is_nodule = prob >= threshold
    bg_color = "#0f172a"
    accent = "#ef4444" if is_nodule else "#22c55e"
    label = "🔴  KNOTEN ERKANNT" if is_nodule else "🟢  KEIN KNOTEN"

    fig = plt.figure(figsize=(18, 5), facecolor=bg_color)
    gs = fig.add_gridspec(1, 5, wspace=0.3)

    slices = [
        ("Axial (Mitte)", patch_array[16, :, :]),
        ("Koronal (Mitte)", patch_array[:, 24, :]),
        ("Sagittal (Mitte)", patch_array[:, :, 24]),
    ]

    for idx, (title, sl) in enumerate(slices):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(bg_color)
        ax.imshow(sl, cmap="bone", aspect="auto")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")

    # Wahrscheinlichkeitsbalken
    ax_bar = fig.add_subplot(gs[0, 3:])
    ax_bar.set_facecolor("#1e293b")

    bar_height = 0.5
    ax_bar.barh([0], [prob], color=accent, height=bar_height, zorder=3)
    ax_bar.barh([0], [1 - prob], left=[prob], color="#334155", height=bar_height, zorder=2)
    ax_bar.axvline(x=threshold, color="#facc15", linewidth=2.5,
                   linestyle="--", label=f"Threshold: {threshold:.2f}", zorder=4)

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.8, 0.8)
    ax_bar.set_xlabel("Wahrscheinlichkeit", color="white", fontsize=11)
    ax_bar.tick_params(colors="white", labelsize=10)
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")

    ax_bar.set_title(label, color=accent, fontsize=16, fontweight="bold", pad=14)
    ax_bar.text(0.5, 0, f"{prob * 100:.1f}%",
                ha="center", va="center", color="white",
                fontsize=20, fontweight="bold", transform=ax_bar.transAxes, zorder=5)
    ax_bar.legend(facecolor="#1e293b", labelcolor="#facc15",
                  fontsize=10, loc="lower right", framealpha=0.8)

    fig.suptitle("CT-Patch — Axial | Koronal | Sagittal",
                 color="#94a3b8", fontsize=11, y=1.01)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=bg_color, dpi=130)
    plt.close(fig)
    return buf_to_pil(buf)


def classify_demo(demo_choice, threshold):
    # WICHTIG: "Knoten-Beispiel" ist ein Substring von "Nicht-Knoten-Beispiel"!
    # Daher prüfen wir auf das eindeutige rote Emoji 🔴 (nur beim Knoten-Beispiel).
    patch = DEMO_NODULE.copy() if "\U0001f534" in demo_choice else DEMO_NON_NODULE.copy()
    prob = predict_patch(patch)
    is_nodule = prob >= threshold
    label = "🔴 KNOTEN erkannt" if is_nodule else "🟢 Kein Knoten"

    summary = f"""## {label}

| Metrik | Wert |
|--------|------|
| **Wahrscheinlichkeit** | `{prob * 100:.2f}%` |
| **Threshold** | `{threshold:.2f}` |
| **Ergebnis** | {"Knoten ⚠️" if is_nodule else "Kein Knoten ✅"} |

> *Synthetischer Demo-Patch — für klinischen Einsatz echte LUNA16-Daten verwenden.*
"""
    img = visualize_patch(patch, prob, threshold)
    return img, summary


def classify_npy(npy_file, threshold):
    if npy_file is None:
        return None, "⚠️ Bitte eine .npy-Datei hochladen."
    try:
        patch = np.load(npy_file.name).astype(np.float32)
    except Exception as e:
        return None, f"❌ Fehler beim Laden: {e}"

    if patch.shape != (32, 48, 48):
        if patch.size == 32 * 48 * 48:
            patch = patch.reshape(32, 48, 48)
        else:
            return None, f"❌ Falsche Form: {patch.shape}. Erwartet: (32, 48, 48)"

    prob = predict_patch(patch)
    is_nodule = prob >= threshold
    label = "🔴 KNOTEN erkannt" if is_nodule else "🟢 Kein Knoten"

    summary = f"""## {label}

| Metrik | Wert |
|--------|------|
| **Wahrscheinlichkeit** | `{prob * 100:.2f}%` |
| **Threshold** | `{threshold:.2f}` |
| **Ergebnis** | {"Knoten ⚠️" if is_nodule else "Kein Knoten ✅"} |
"""
    img = visualize_patch(patch, prob, threshold)
    return img, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
css = """
/* ---- Dunkler Hintergrund für die gesamte App ---- */
body, .gradio-container {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
}

/* ---- Überschriften ---- */
h1 { font-size: 2rem !important; font-weight: 800 !important; color: #f8fafc !important; }
h2 { font-size: 1.4rem !important; color: #f1f5f9 !important; }
h3 { color: #f1f5f9 !important; }

/* ---- Fließtext, Labels, Markdown ---- */
p, label, .label-wrap span, span, li {
    color: #cbd5e1 !important;
    font-size: 1rem !important;
}
.prose p, .prose li, .prose strong { color: #cbd5e1 !important; }

/* ---- Tabellen in Markdown ---- */
table { color: #e2e8f0 !important; }
th { color: #f8fafc !important; background-color: #1e293b !important; }
td { color: #cbd5e1 !important; background-color: #1e293b !important; }

/* ---- Panels / Boxes ---- */
.block, .panel, .form {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* ---- Tabs ---- */
.tab-nav { background-color: #1e293b !important; border-bottom: 1px solid #334155 !important; }
.tab-nav button {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
}
.tab-nav button.selected { color: #f97316 !important; border-bottom: 2px solid #f97316 !important; }

/* ---- Radio buttons ---- */
.radio-group label { color: #e2e8f0 !important; }

/* ---- Slider ---- */
.slider input { accent-color: #f97316 !important; }

/* ---- Code blocks ---- */
code, pre { background-color: #0f172a !important; color: #7dd3fc !important; border-radius: 4px !important; }

/* ---- Container ---- */
.gradio-container { max-width: 960px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    css=css,
    title="🫁 Lung Cancer Detection",
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="sky",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        text_size=gr.themes.sizes.text_md,
    ),
) as demo:

    gr.Markdown("""
# 🫁 Lungenknoten-Klassifizierung mit 3D CNN

**Trainiert auf LUNA16 &nbsp;|&nbsp; AUC = 0.987 &nbsp;|&nbsp; Recall = 93 %**

Dieses Modell klassifiziert 3D-CT-Patches (32 × 48 × 48 Voxel) als **Knoten** oder **Nicht-Knoten**.
""")

    with gr.Tabs():

        # --- Tab 1: Demo ---
        with gr.TabItem("🎯 Demo (sofort ausprobieren)"):
            gr.Markdown("### Wähle einen vorbereiteten Beispiel-Patch aus und klicke auf Klassifizieren.")
            with gr.Row():
                demo_choice = gr.Radio(
                    choices=["🔴 Knoten-Beispiel (simuliert)",
                             "🟢 Nicht-Knoten-Beispiel (simuliert)"],
                    value="🔴 Knoten-Beispiel (simuliert)",
                    label="Beispiel",
                )
                threshold_demo = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Klassifizierungs-Threshold (0.5 = Standard)",
                )
            btn_demo = gr.Button("▶  Klassifizieren", variant="primary", size="lg")
            with gr.Row():
                img_demo = gr.Image(label="CT-Visualisierung", type="pil",
                                    show_label=True)
                result_demo = gr.Markdown(value="*Klicke auf 'Klassifizieren' um das Ergebnis zu sehen.*")
            btn_demo.click(classify_demo,
                           inputs=[demo_choice, threshold_demo],
                           outputs=[img_demo, result_demo])

        # --- Tab 2: Eigene Datei ---
        with gr.TabItem("📂 Eigene .npy-Datei"):
            gr.Markdown("""
### Lade deinen eigenen CT-Patch hoch

**Erwartete Form:** `(32, 48, 48)` — Voxelwerte in Hounsfield-Einheiten (HU)

```python
import numpy as np
# patch_array shape: (32, 48, 48) aus deinem CT-Scan
np.save("mein_patch.npy", patch_array)
```
""")
            with gr.Row():
                npy_file = gr.File(label="CT-Patch (.npy)", file_types=[".npy"])
                threshold_upload = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Klassifizierungs-Threshold",
                )
            btn_upload = gr.Button("▶  Klassifizieren", variant="primary", size="lg")
            with gr.Row():
                img_upload = gr.Image(label="CT-Visualisierung", type="pil")
                result_upload = gr.Markdown()
            btn_upload.click(classify_npy,
                             inputs=[npy_file, threshold_upload],
                             outputs=[img_upload, result_upload])

    gr.Markdown("""
---
**Modell:** 3D-CNN · 4 LunaBlocks (Conv3D + BatchNorm + MaxPool) &nbsp;|&nbsp;
**Datensatz:** LUNA16 (~551 K Kandidaten, 0,25 % echte Knoten) &nbsp;|&nbsp;
**Training:** 20 Epochen GPU (Google Colab) &nbsp;|&nbsp;
[📁 GitHub](https://github.com/awildt01/deep-learning-lung-cancer-detection)
""")

demo.launch()
