"""
Example script demonstrating how to use the Brain Tumor Segmentation API.
"""

import requests
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"


def upload_model(model_path: str):
    """Upload a model weights file."""
    print(f"Uploading model from {model_path}...")
    
    with open(model_path, "rb") as f:
        files = {"file": (os.path.basename(model_path), f, "application/octet-stream")}
        response = requests.post(f"{BASE_URL}/upload-model", files=files)
    
    if response.status_code == 200:
        print("✓ Model uploaded successfully!")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Error uploading model: {response.status_code}")
        print(f"  {response.text}")
    
    return response


def inference_image(image_path: str):
    """Perform inference on a grayscale image."""
    print(f"\nPerforming inference on image: {image_path}...")
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        response = requests.post(f"{BASE_URL}/inference/image", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Inference completed successfully!")
        print(f"  Shape: {result['shape']}")
        print(f"  Classes detected: {result['classes_detected']}")
        print(f"  Output file: {result.get('output_path', 'N/A')}")
        
        # Download the result if available
        if result.get('output_path'):
            download_segmentation(result['output_path'])
    else:
        print(f"✗ Error during inference: {response.status_code}")
        print(f"  {response.text}")
    
    return response


def inference_nifti(nifti_path: str, endpoint: str = "nifti"):
    """Perform inference on a NIfTI file."""
    print(f"\nPerforming inference on NIfTI file: {nifti_path}...")
    
    with open(nifti_path, "rb") as f:
        files = {"file": (os.path.basename(nifti_path), f, "application/octet-stream")}
        response = requests.post(f"{BASE_URL}/inference/{endpoint}", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Inference completed successfully!")
        print(f"  Shape: {result['shape']}")
        print(f"  Classes detected: {result['classes_detected']}")
        print(f"  Output file: {result.get('output_path', 'N/A')}")
        
        # Download the result if available
        if result.get('output_path'):
            download_segmentation(result['output_path'])
    else:
        print(f"✗ Error during inference: {response.status_code}")
        print(f"  {response.text}")
    
    return response


def download_segmentation(filename: str):
    """Download a segmentation result file."""
    print(f"\nDownloading segmentation: {filename}...")
    
    response = requests.get(f"{BASE_URL}/download-segmentation/{filename}")
    
    if response.status_code == 200:
        output_path = Path("downloads") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        print(f"✓ Segmentation saved to: {output_path}")
    else:
        print(f"✗ Error downloading file: {response.status_code}")
    
    return response


def check_health():
    """Check API health status."""
    print("Checking API health...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print("✓ API is healthy!")
        print(f"  Status: {result['status']}")
        print(f"  Model loaded: {result['model_loaded']}")
        print(f"  Device: {result['device']}")
    else:
        print(f"✗ API health check failed: {response.status_code}")
    
    return response


if __name__ == "__main__":
    print("=" * 60)
    print("Brain Tumor Segmentation API - Example Usage")
    print("=" * 60)
    
    # Check health
    check_health()
    
    # Example: Upload model (uncomment and provide path)
    # model_path = "model_weights/Entire Model.pt"
    # if os.path.exists(model_path):
    #     upload_model(model_path)
    
    # Example: Inference on image (uncomment and provide path)
    # image_path = "path/to/your/image.png"
    # if os.path.exists(image_path):
    #     inference_image(image_path)
    
    # Example: Inference on NIfTI (uncomment and provide path)
    # nifti_path = "path/to/your/volume.nii.gz"
    # if os.path.exists(nifti_path):
    #     inference_nifti(nifti_path, endpoint="nifti")
    #     # Or use specialized endpoints:
    #     # inference_nifti(nifti_path, endpoint="nifti/flair")
    #     # inference_nifti(nifti_path, endpoint="nifti/t1ce")
    
    print("\n" + "=" * 60)
    print("Example completed. Uncomment sections above to test with your files.")
    print("=" * 60)
