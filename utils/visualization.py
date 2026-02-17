import numpy as np
import cv2
import math
import base64

SEG_COLORS = np.array([
    [0,   0,   0],    # 0 – background
    [255, 0,   0],    # 1 – necrotic core
    [0,   255, 0],    # 2 – edema
    [0,   0,   255],  # 3 – enhancing tumor
], dtype=np.uint8)


def build_overlay_grid(
    flair_volume: np.ndarray,       # (H, W, D)  raw FLAIR voxels
    seg_volume: np.ndarray,         # (H, W, D)  uint8 class labels
    img_size: int = 128,
    alpha: float = 0.5,
    max_slices: int = 100,
    cols: int = 10,
) -> bytes:
    """
    Combine every axial slice into a single PNG grid image.
    Each cell shows the FLAIR slice with the segmentation mask overlaid
    in the same colour scheme as showPredictsById.

    Returns the PNG as raw bytes.
    """
    depth = seg_volume.shape[2]
    # Evenly sample up to max_slices axial slices across the volume
    indices = np.linspace(0, depth - 1, min(max_slices, depth), dtype=int)

    rows = math.ceil(len(indices) / cols)
    cell_h, cell_w = img_size, img_size
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for pos, idx in enumerate(indices):
        row, col = divmod(pos, cols)
        y0, y1 = row * cell_h, (row + 1) * cell_h
        x0, x1 = col * cell_w, (col + 1) * cell_w

        # ── FLAIR background ──────────────────────────────────────────────
        flair_slice = flair_volume[:, :, idx]
        flair_resized = cv2.resize(flair_slice, (cell_w, cell_h)).astype(np.float32)
        flair_norm = flair_resized / (flair_resized.max() + 1e-8)
        flair_gray = (flair_norm * 255).astype(np.uint8)
        flair_rgb = cv2.cvtColor(flair_gray, cv2.COLOR_GRAY2RGB)

        # ── Segmentation overlay ──────────────────────────────────────────
        seg_slice = seg_volume[:, :, idx].astype(np.uint8)
        seg_resized = cv2.resize(seg_slice, (cell_w, cell_h),
                                 interpolation=cv2.INTER_NEAREST)
        seg_clipped = np.clip(seg_resized, 0, len(SEG_COLORS) - 1)
        seg_rgb = SEG_COLORS[seg_clipped]          # (H, W, 3)

        # Blend only where a tumour class is present (skip pure background)
        tumor_mask = (seg_clipped > 0)[:, :, np.newaxis]
        cell = np.where(
            tumor_mask,
            ((1 - alpha) * flair_rgb + alpha * seg_rgb).astype(np.uint8),
            flair_rgb,
        )
        canvas[y0:y1, x0:x1] = cell

    # Encode canvas → PNG bytes
    success, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("cv2.imencode failed to produce PNG bytes")
    return buf.tobytes()