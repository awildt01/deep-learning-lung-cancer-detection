"""Gradio Demo – Lung Cancer Nodule Classifier (3D CNN, LUNA16)."""

import io

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
CHECKPOINT = "luna_model_best.pt"
DEVICE = torch.device("cpu")


def load_model():
    model = LunaModel()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
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
    patch = DEMO_NODULE.copy() if "Knoten-Beispiel" in demo_choice else DEMO_NON_NODULE.copy()
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
h1 { font-size: 2rem !important; font-weight: 800 !important; color: #f8fafc !important; }
h2 { font-size: 1.4rem !important; color: #e2e8f0 !important; }
p, label { color: #cbd5e1 !important; font-size: 1rem !important; }
.gradio-container { max-width: 960px !important; margin: auto !important; }
.tab-nav button { font-size: 1rem !important; font-weight: 600 !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    css=css,
    title="🫁 Lung Cancer Detection",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
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
