# Brain Tumor Segmentation & Classification API

FastAPI-based service for brain tumor segmentation (U-Net) and neurological disorder classification (SMG-Net). This service provides endpoints for model uploading and inference on various MRI image formats.

## Features

### Segmentation (U-Net)
- **Model Upload**: Upload and load PyTorch model weights (.pt, .pth files)
- **Grayscale Image Inference**: Process single grayscale images (PNG, JPEG, TIFF)
- **NIfTI File Inference**: Process 3D NIfTI volumes (.nii, .nii.gz)
- **FLAIR NIfTI Inference**: Specialized endpoint for FLAIR MRI sequences
- **T1CE NIfTI Inference**: Specialized endpoint for T1-weighted contrast-enhanced MRI sequences

### Classification (SMG-Net)
- **SMG-Net Model Upload**: Upload and load SMG-Net classification model weights
- **Single Image Classification**: Classify a single MRI image into neurological disorder categories
- **Batch Classification**: Classify multiple MRI images in a single request
- **Default Classes**: Alzheimer's, MS, Normal, Tumor (customizable per request)

## Installation

### Prerequisites

- Python 3.10 or higher (Python 3.11+ recommended for latest package versions)
- pip (Python package installer)

### Setup Steps

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

   Or on some systems:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**

   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

   On Windows:
   ```bash
   venv\Scripts\activate
   ```

3. **Upgrade pip, setuptools, and wheel:**

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

   This ensures you have the latest build tools needed for installing packages.

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure you have model weights:**

   Place your model weights (.pt or .pth files) in the `model_weights/` directory, or upload them via the API after starting the server.

### Deactivating the Virtual Environment

When you're done working, you can deactivate the virtual environment:

```bash
deactivate
```

## Usage

### Start the server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

#### Health Check
- `GET /` - Root endpoint with health status
- `GET /health` - Health check endpoint

#### Model Management
- `POST /upload-model` - Upload a new segmentation model weights file
- `POST /upload-model/smgnet` - Upload a new SMG-Net classification model weights file (query params: `num_classes`, `in_channels`)

#### Segmentation Inference
- `POST /inference/image` - Inference on grayscale image
- `POST /inference/nifti` - Inference on NIfTI file; returns a 3D NIfTI mask (`nifti_path`) and a PNG visualization (`output_path`)
- `POST /inference/nifti/flair` - Inference on FLAIR NIfTI file (same response structure as `/inference/nifti`)
- `POST /inference/nifti/t1ce` - Inference on T1CE NIfTI file (same response structure as `/inference/nifti`)

> **InferenceResponse fields**
> - `status`: `"success"` or error status
> - `message`: human-readable description
> - `shape`: output tensor/volume shape
> - `classes_detected`: list of class IDs present in the segmentation
> - `output_format`: `"image"` for PNG visualizations, `"nifti"` for raw volumes (image endpoint uses `"image"`)
> - `output_path`: filename of the main visualization image (PNG) stored in `uploaded_models/`
> - `nifti_path`: filename of the full 3D NIfTI segmentation volume (for NIfTI endpoints)

#### Classification (SMG-Net)
- `POST /classification/classify_image` - Classify a single MRI image
- `POST /classification/classify_batch` - Classify multiple MRI images in batch

> **ClassificationResponse fields**
> - `status`: `"success"` or `"error"`
> - `message`: human-readable description
> - `predicted_class`: integer class index
> - `predicted_class_name`: human-readable class label (e.g. `"Alzheimer's"`)
> - `class_probabilities`: list of probabilities for each class
> - `class_names`: list of all class labels
> - `confidence`: probability of the predicted class

#### Download Results
- `GET /download-segmentation/{filename}` - Download segmentation result file

### Viewing segmentation results

Results are saved under `uploaded_models/` (e.g. `segmentation_20260215_122803.nii.gz` or `.png`).

**Option 1 — Download via API**

```bash
# Use the filename returned in the inference response (output_path)
curl -O -J "http://localhost:8000/download-segmentation/segmentation_20260215_122803.nii.gz"
```

**Option 2 — View locally with the script**

List recent results and open a file (NIfTI slice viewer or PNG in default viewer):

```bash
# List segmentation files in uploaded_models
python scripts/view_segmentation.py

# View a NIfTI volume (slice slider)
python scripts/view_segmentation.py segmentation_20260215_122803.nii.gz

# View a PNG result
python scripts/view_segmentation.py segmentation_20260215_122803.png
```

Requires: `nibabel`, `matplotlib` (in `requirements.txt`).

### Example Usage

#### Upload Model:
```bash
curl -X POST "http://localhost:8000/upload-model" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@model_weights/Entire Model.pt"
```

#### Inference on Image:
```bash
curl -X POST "http://localhost:8000/inference/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.png"
```

#### Inference on NIfTI:
```bash
curl -X POST "http://localhost:8000/inference/nifti" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/volume.nii.gz"
```

#### Upload SMG-Net Model:
```bash
curl -X POST "http://localhost:8000/upload-model/smgnet?num_classes=4&in_channels=1" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@model_weights/smgnet_best.pth"
```

#### Classify Single Image:
```bash
curl -X POST "http://localhost:8000/classification/classify_image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/mri_image.png" \
  -F "class_names=Alzheimer's, MS, Normal, Tumor"
```

#### Classify Batch of Images:
```bash
curl -X POST "http://localhost:8000/classification/classify_batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@path/to/image1.png" \
  -F "files=@path/to/image2.png" \
  -F "class_names=Alzheimer's, MS, Normal, Tumor"
```

## Model Architectures

### U-Net (Segmentation)
- Input: 1 channel (grayscale), 256x256 pixels
- Output: 4 classes (background, edema, non-enhancing tumor, enhancing tumor)
- Default weights: `model_weights/Entire Model.pt`

### SMG-Net (Classification)
- Input: 1 channel (grayscale), 512x512 pixels
- Output: 4 classes (Alzheimer's, MS, Normal, Tumor) — customizable via `class_names` parameter
- Architecture: CNN backbone (conv layers with BN + ReLU + MaxPool) → global average pooling → dropout (0.5) → FC
- Default weights: `model_weights/smgnet_best.pth`

## Project Structure

```
fastapi_ai_model_service/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration settings
├── requirements.txt        # Python dependencies
├── schemas.py             # Pydantic models
├── api/                   # API endpoints (modular routers)
│   ├── __init__.py
│   ├── health.py         # Health check endpoints
│   ├── model.py          # Model upload endpoints (segmentation + SMG-Net)
│   ├── inference.py      # Segmentation inference endpoints
│   └── classification.py # SMG-Net classification endpoints
├── core/                  # Core services
│   ├── __init__.py
│   ├── model_service.py  # Singleton segmentation model service
│   └── smgnet_service.py # Singleton SMG-Net classification service
├── models/                # Model architectures
│   ├── __init__.py
│   ├── unet.py           # U-Net model architecture
│   └── smgnet.py         # SMG-Net classification architecture
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py  # Image preprocessing utilities
│   └── postprocessing.py # Segmentation postprocessing
├── model_weights/         # Default model weights directory
└── uploaded_models/       # Uploaded models and results directory
```

## Architecture

The application follows a modular architecture:

- **`main.py`**: Minimal entry point that initializes FastAPI app and includes routers
- **`config.py`**: Centralized configuration (paths, device settings, model parameters)
- **`core/model_service.py`**: Singleton pattern for segmentation model loading and management
- **`core/smgnet_service.py`**: Singleton pattern for SMG-Net classification model loading and management
- **`api/`**: Separated endpoint routers for better organization
  - `health.py`: Health check endpoints
  - `model.py`: Model upload and management (both segmentation and classification)
  - `inference.py`: Segmentation inference endpoints
  - `classification.py`: SMG-Net classification endpoints

## Notes

- Both models automatically load from `model_weights/` directory on startup (U-Net segmentation + SMG-Net classification)
- GPU support is automatic if CUDA is available
- Segmentation results are temporarily stored in `uploaded_models/` directory
- Segmentation input images are automatically resized to 256x256; classification inputs are resized to 512x512
- NIfTI volumes are processed slice-by-slice
- Classification `class_names` parameter is optional — defaults to `"Alzheimer's, MS, Normal, Tumor"`
