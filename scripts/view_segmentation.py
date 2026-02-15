#!/usr/bin/env python3
"""
View segmentation results from uploaded_models directory.

Usage:
  python scripts/view_segmentation.py                    # list recent results
  python scripts/view_segmentation.py segmentation_20260215_122803.nii.gz   # view NIfTI
  python scripts/view_segmentation.py segmentation_20260215_122803.png     # view PNG

For NIfTI: use arrow keys or slider to move through slices. Close window to exit.
"""

import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import UPLOADED_MODELS_DIR


def list_results():
    """List segmentation files in uploaded_models, newest first."""
    if not UPLOADED_MODELS_DIR.exists():
        print(f"Directory not found: {UPLOADED_MODELS_DIR}")
        return
    files = sorted(
        [f for f in UPLOADED_MODELS_DIR.iterdir() if f.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        print("No segmentation results in uploaded_models/")
        return
    print("Segmentation results (newest first):")
    for f in files:
        print(f"  {f.name}")
    print("\nView a file:  python scripts/view_segmentation.py <filename>")


def view_nifti(filepath: Path):
    """Display NIfTI segmentation volume slice-by-slice with matplotlib."""
    try:
        import nibabel as nib
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError as e:
        print("Install: pip install nibabel matplotlib numpy")
        raise SystemExit(1) from e

    img = nib.load(str(filepath))
    data = np.asarray(img.get_fdata())
    # Assume shape like (H, W, D) for slices along last axis
    if data.ndim != 3:
        print(f"Expected 3D volume, got shape {data.shape}")
        return

    n_slices = data.shape[2]
    vmin, vmax = 0, int(data.max()) if data.size else 1

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    slice_ax = ax.imshow(data[:, :, 0], cmap="nipy_spectral", vmin=vmin, vmax=vmax)
    ax.set_title(f"{filepath.name} — slice 0 / {n_slices}")
    ax.axis("off")

    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, "Slice", 0, n_slices - 1, valinit=0, valstep=1)

    def update(_):
        i = int(slider.val)
        slice_ax.set_data(data[:, :, i])
        ax.set_title(f"{filepath.name} — slice {i} / {n_slices}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def view_png(filepath: Path):
    """Open PNG segmentation in default viewer."""
    try:
        from PIL import Image
        img = Image.open(filepath)
        img.show()
    except Exception as e:
        print(f"Cannot display image: {e}")


def main():
    if len(sys.argv) < 2:
        list_results()
        return

    filename = sys.argv[1]
    filepath = UPLOADED_MODELS_DIR / filename
    if not filepath.exists():
        print(f"File not found: {filepath}")
        list_results()
        return

    suffix = filepath.suffix.lower()
    if filename.endswith(".nii.gz"):
        view_nifti(filepath)
    elif suffix == ".nii":
        view_nifti(filepath)
    elif suffix in (".png", ".jpg", ".jpeg"):
        view_png(filepath)
    else:
        print(f"Unknown format: {filename}. Use .nii, .nii.gz, or .png")


if __name__ == "__main__":
    main()
