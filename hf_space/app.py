"""Gradio Demo – Lung Cancer Nodule Classifier (3D CNN, LUNA16)."""

import io
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    prob_nodule = float(probs[0, 1])
    return prob_nodule


def visualize_patch(patch_array: np.ndarray, prob: float, threshold: float):
    """Erstellt eine Visualisierung mit 3 Schichten und Wahrscheinlichkeitsbalken."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                              facecolor="#0f172a")

    slices = [
        ("Axial\n(Mitte)", patch_array[16, :, :]),
        ("Koronal\n(Mitte)", patch_array[:, 24, :]),
        ("Sagittal\n(Mitte)", patch_array[:, :, 24]),
    ]

    for ax, (title, sl) in zip(axes[:3], slices):
        ax.imshow(sl, cmap="bone", aspect="auto")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")

    # Wahrscheinlichkeitsbalken
    ax_bar = axes[3]
    ax_bar.set_facecolor("#1e293b")
    bar_color = "#ef4444" if prob >= threshold else "#22c55e"
    label = "🔴 KNOTEN" if prob >= threshold else "🟢 KEIN KNOTEN"

    ax_bar.barh([0], [prob], color=bar_color, height=0.4)
    ax_bar.barh([0], [1 - prob], left=[prob], color="#334155", height=0.4)
    ax_bar.axvline(x=threshold, color="#facc15", linewidth=2,
                   linestyle="--", label=f"Threshold: {threshold:.2f}")

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.set_xlabel("Wahrscheinlichkeit", color="white", fontsize=10)
    ax_bar.set_title(label, color=bar_color, fontsize=13, fontweight="bold", pad=10)
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#334155")
    ax_bar.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)

    prob_text = f"{prob * 100:.1f}%"
    ax_bar.text(0.5, 0, prob_text, ha="center", va="center",
                color="white", fontsize=16, fontweight="bold",
                transform=ax_bar.transAxes)

    fig.suptitle("Lungenknoten-Klassifizierung — 3D CNN (LUNA16)",
                 color="white", fontsize=13, y=1.02)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0f172a", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def classify_npy(npy_file, threshold):
    """Verarbeitet eine hochgeladene .npy-Datei."""
    if npy_file is None:
        return None, "⚠️ Bitte eine .npy-Datei hochladen."

    try:
        patch = np.load(npy_file.name)
    except Exception as e:
        return None, f"❌ Fehler beim Laden: {e}"

    patch = patch.astype(np.float32)
    if patch.shape != (32, 48, 48):
        if patch.size == 32 * 48 * 48:
            patch = patch.reshape(32, 48, 48)
        else:
            return None, f"❌ Falsche Form: {patch.shape}. Erwartet: (32, 48, 48)"

    prob = predict_patch(patch)
    label = "🔴 KNOTEN erkannt" if prob >= threshold else "🟢 Kein Knoten"
    summary = (f"**Ergebnis:** {label}\n\n"
               f"**Wahrscheinlichkeit:** `{prob * 100:.2f}%`\n\n"
               f"**Threshold:** `{threshold:.2f}`")

    buf = visualize_patch(patch, prob, threshold)
    img = plt.imread(buf)
    return img, summary


def classify_demo(demo_choice, threshold):
    """Verwendet einen der vorbereiteten Demo-Patches."""
    if demo_choice == "🔴 Knoten-Beispiel (simuliert)":
        patch = DEMO_NODULE.copy()
    else:
        patch = DEMO_NON_NODULE.copy()

    prob = predict_patch(patch)
    label = "🔴 KNOTEN erkannt" if prob >= threshold else "🟢 Kein Knoten"
    summary = (f"**Ergebnis:** {label}\n\n"
               f"**Wahrscheinlichkeit:** `{prob * 100:.2f}%`\n\n"
               f"**Threshold:** `{threshold:.2f}`\n\n"
               f"*Hinweis: Dies ist ein synthetischer Demo-Patch.*")

    buf = visualize_patch(patch, prob, threshold)
    img = plt.imread(buf)
    return img, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
css = """
body { background: #0f172a; }
.gradio-container { max-width: 900px; margin: auto; }
"""

with gr.Blocks(css=css, title="🫁 Lung Cancer Detection") as demo:

    gr.Markdown("""
    # 🫁 Lungenknoten-Klassifizierung mit 3D CNN
    **Trainiert auf LUNA16 | AUC = 0.987 | Recall = 93%**

    Dieses Modell klassifiziert 3D-CT-Patches (32×48×48 Voxel) als **Knoten** oder **Nicht-Knoten**.
    """)

    with gr.Tabs():

        # Tab 1: Demo
        with gr.TabItem("🎯 Demo (sofort ausprobieren)"):
            gr.Markdown("Wähle einen vorbereiteten Beispiel-Patch aus:")
            with gr.Row():
                demo_choice = gr.Radio(
                    choices=["🔴 Knoten-Beispiel (simuliert)",
                             "🟢 Nicht-Knoten-Beispiel (simuliert)"],
                    value="🔴 Knoten-Beispiel (simuliert)",
                    label="Beispiel auswählen",
                )
                threshold_demo = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Klassifizierungs-Threshold",
                )
            btn_demo = gr.Button("▶ Klassifizieren", variant="primary")
            with gr.Row():
                img_demo = gr.Image(label="Visualisierung", type="numpy")
                result_demo = gr.Markdown()
            btn_demo.click(classify_demo,
                           inputs=[demo_choice, threshold_demo],
                           outputs=[img_demo, result_demo])

        # Tab 2: Eigene Datei
        with gr.TabItem("📂 Eigene .npy-Datei"):
            gr.Markdown("""
            Lade einen eigenen CT-Patch im `.npy`-Format hoch.
            **Erwartete Form:** `(32, 48, 48)` — Voxelwerte in Hounsfield-Einheiten (HU).

            Patch aus LUNA16 extrahieren:
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
            btn_upload = gr.Button("▶ Klassifizieren", variant="primary")
            with gr.Row():
                img_upload = gr.Image(label="Visualisierung", type="numpy")
                result_upload = gr.Markdown()
            btn_upload.click(classify_npy,
                             inputs=[npy_file, threshold_upload],
                             outputs=[img_upload, result_upload])

    gr.Markdown("""
    ---
    **Modell:** 3D-CNN mit 4 LunaBlocks (Conv3D + BatchNorm + MaxPool) · 
    **Datensatz:** LUNA16 (~551.000 Kandidaten, 0,25% echte Knoten) · 
    **Training:** 20 Epochen auf GPU (Google Colab) · 
    [GitHub](https://github.com/awildt01/deep-learning-lung-cancer-detection)
    """)

demo.launch()
