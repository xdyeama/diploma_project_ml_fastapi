import pydicom
import numpy as np
import os
import nibabel as nib
import cv2
import math
# ============================================================================
# HELPER
# ============================================================================

def _load_dicom_volume(dcm_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one or more DICOM files into a single (H, W, D) volume.

    - Single file  → (H, W, 1)  for 2-D, or (H, W, frames) for multi-frame
    - Multiple files → sorted by InstanceNumber (or filename), stacked along D

    Returns:
        volume  – float32 array (H, W, D) with rescale slope/intercept applied
        affine  – 4×4 affine matrix built from DICOM spatial metadata
    """
    slices = []
    for path in dcm_paths:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)

        slope     = float(getattr(ds, "RescaleSlope",     1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        arr = arr * slope + intercept

        if arr.ndim == 2:
            # Single-frame: (H, W) → treat as one slice
            slices.append((ds, arr[:, :, np.newaxis]))
        elif arr.ndim == 3:
            # Multi-frame DICOM: (frames, H, W) → (H, W, frames)
            slices.append((ds, np.transpose(arr, (1, 2, 0))))

    # Sort by InstanceNumber so slices are in anatomical order
    slices.sort(key=lambda x: int(getattr(x[0], "InstanceNumber", 0)))

    # Stack all slices along depth axis
    volume = np.concatenate([s[1] for s in slices], axis=2)  # (H, W, D)

    # Build affine from the first slice's spatial metadata
    try:
        ds_ref         = slices[0][0]
        pixel_spacing  = [float(v) for v in ds_ref.PixelSpacing]
        slice_thickness = float(getattr(ds_ref, "SliceThickness", 1.0))
        affine = np.diag([pixel_spacing[1], pixel_spacing[0], slice_thickness, 1.0])
    except Exception:
        affine = np.eye(4)

    return volume, affine


def _crop_black_borders(image: np.ndarray) -> np.ndarray:
    """
    Crop away fully-black rows and columns from an RGB image (H, W, 3).
    Falls back to the original image if nothing is non-black.
    """
    gray = image.sum(axis=2)                        # (H, W) — 0 where all black
    rows = np.any(gray > 0, axis=1)
    cols = np.any(gray > 0, axis=0)

    if not rows.any() or not cols.any():
        return image                                # nothing to crop

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return image[r0:r1 + 1, c0:c1 + 1]


def _build_overlay_grid(
    flair_volume: np.ndarray,       # (H, W, D) raw voxels
    seg_volume:   np.ndarray,       # (H, W, D) uint8 class labels
    img_size:  int   = 128,
    alpha:     float = 0.5,
    max_slices: int  = 100,
    cols:       int  = 10,
) -> bytes:
    """
    Combine axial slices into a single PNG grid with FLAIR + segmentation
    overlay, then crop away any trailing black cells.

    Colour scheme (matches showPredictsById):
        red   → necrotic core  (class 1)
        green → edema          (class 2)
        blue  → enhancing      (class 3)
    """
    SEG_COLORS = np.array([
        [0,   0,   0],    # 0 – background
        [255, 0,   0],    # 1 – necrotic core
        [0,   255, 0],    # 2 – edema
        [0,   0,   255],  # 3 – enhancing tumor
    ], dtype=np.uint8)

    depth   = seg_volume.shape[2]
    indices = np.linspace(0, depth - 1, min(max_slices, depth), dtype=int)

    rows_count = math.ceil(len(indices) / cols)
    cell_h, cell_w = img_size, img_size
    canvas = np.zeros((rows_count * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for pos, idx in enumerate(indices):
        row, col = divmod(pos, cols)
        y0, y1   = row * cell_h, (row + 1) * cell_h
        x0, x1   = col * cell_w, (col + 1) * cell_w

        # FLAIR background
        flair_slice = flair_volume[:, :, idx]
        flair_resized = cv2.resize(flair_slice, (cell_w, cell_h)).astype(np.float32)
        flair_norm    = flair_resized / (flair_resized.max() + 1e-8)
        flair_rgb     = cv2.cvtColor((flair_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Segmentation overlay
        seg_slice   = seg_volume[:, :, idx].astype(np.uint8)
        seg_resized = cv2.resize(seg_slice, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        seg_clipped = np.clip(seg_resized, 0, len(SEG_COLORS) - 1)
        seg_rgb     = SEG_COLORS[seg_clipped]

        tumor_mask = (seg_clipped > 0)[:, :, np.newaxis]
        cell = np.where(
            tumor_mask,
            ((1 - alpha) * flair_rgb + alpha * seg_rgb).astype(np.uint8),
            flair_rgb,
        )
        canvas[y0:y1, x0:x1] = cell

    # Crop trailing black cells from the last row
    canvas = _crop_black_borders(canvas)

    success, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("cv2.imencode failed to produce PNG bytes")
    return buf.tobytes()