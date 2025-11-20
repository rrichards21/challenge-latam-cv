# Machine Learning Engineer Challenge

## Overview

This document summarizes the development and operationalization process for the challenge about object detection system, detailing data analysis, architecture choices and the solutions implemented to overcome technical roadblocks encountered during development, training, and API integration.

## Part I: Model Training and Artifact Generation

The goal was to train a YOLO-based model with a given dataset and export it as inference-ready artifacts.

1. Data Analysis and Preparation

The given jupyter notebook (exploration.ipynb) had data exploration steps already outlined. Key observations are included in the notebook, but highlights include:
- Imbalanced class distribution, with "Forklift" and "Person" being the majority classes ~94% of the dataset.
- Small object sizes, especially for "Helmet" and "Cone", which may require careful augmentation and model tuning.
- Median and mean bounding box sizes were small relative to image dimensions, indicating the need for high-resolution inputs.

2. Training Setup and Hyperparameters

- Model: YOLO11n pre-trained weights for transfer learning.
- Image Size: 640x640 pixels. Trade-off between VRAM limits and small object detection.
- Batch Size: 16 (limited by GPU memory). Ensured Batch Normalization stability.
- Epochs: 30. Enough to converge given dataset size, complexity and nano backbone.
- Augmentations: Enabled MixUp at 10% to act as a regularizer against class imbalance.

3. Training Results and Evaluation

As stated at the data exploration notebook, the model achieved the predicted metrics as follows:
- High accuracy on majority classes ("Forklift" and "Person") with precision and recall above 90%.
- Lower performance on minority classes ("Helmet" and "Truck"), with recall ~30%, indicating room for improvement.
- Overall mAP@0.5 of approximately 30%, which is reasonable given the dataset challenges, time and hardware limitations.

Faced problems: Artifact Export Instability

Issue: The attempt to export the model to the ONNX format (model.export(format="onnx")) within the local Windows/VS Code Jupyter environment resulted in kernel crashes and silent failures. This prevented the creation of the final model.onnx artifact locally.

Doing a quick search on google and forums revealed that this is a known issue with certain configurations VS Code + Windows + specific PyTorch/ONNX versions, leading to instability during export operations and thus crashing the kernel.

**Solution**:
To work around this, we moved the export step to an isolated python script (export_model.py) and executed it without problems.

## Part II: FastAPI API Implementation

The goal was to implement the /health and /predict endpoints using FastAPI and ensure robustness under test conditions.

**Architectural Choice**: Singleton Pattern

To ensure the model is loaded only once and is accessible across requests, we implemented a singleton pattern using an Object-Oriented Programming (OOP) approach. This encapsulates model loading, preprocessing, inference, and postprocessing logic within a dedicated class (ObjectDetector).

### Error Handling and Logging
During the make api-test phase, I've encountered and resolved some integration errors:

#### Error: 503/501 Service Error

Cause: TestClient executes from the root directory, causing relative paths to fail.

Resolution: Implemented Absolute Path Resolution using Path(__file__).parent to locate artifacts regardless of the Current Working Directory (CWD).

#### Error: 500 Internal Error (DLL Load Failure)

Cause: Incompatibility with onnxruntime on Windows during synchronous initialization.

Resolution: Configured the local environment to load model_best.pt (PyTorch weights) while reserving ONNX for the Linux-based CI/CD pipeline.

#### Error: Test Script Failure (UnicodeDecodeError)

Cause: The test logic failed to swap the .jpg extension for .txt, attempting to read binary image data as text labels.

Resolution: Patched tests/api/test_api_dataset.py to use Path().with_suffix('.txt'), ensuring correct Ground Truth parsing.

## Part III: Deployment in GCP Cloud Run
The API was deployed to GCP Cloud Run using the following steps:

1. Built the Docker image locally using a multi-stage Dockerfile for size optimization.
   - Technique:
    - Discarded build tools (gcc, headers) in the final image.
    - Explicitly utilized torch --extra-index-url .../cpu to exclude CUDA dependencies, as Cloud Run instances are CPU-based.
2. Pushed the image to Google Container Registry (GCR).
3. Deployed the image to Cloud Run, configuring the service to allow unauthenticated access for public API availability.

# Test the challenge API in GCP

In order to test the deployed API, you can use the following `curl` command. Replace `path/to/your/image.jpg` with the actual path to an image file on your local machine that you want to test the object detection on.
```
curl -X POST "https://object-detection-api-255514139094.europe-west1.run.app/predict" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```
This command sends a POST request to the API endpoint with the image file included in the form data. The API should respond with the detection results as follows:
```
{
    "filename": "path/to/your/image.jpg",
    "detections_count": number_of_detections_in_the_image,
    "detections": [
        {
            "class": "detected_object_class_name",
            "class_id": class_id_integer,
            "confidence": float_confidence_score,
            "bbox": [x_min, y_min, x_max, y_max]
        }
    ]
}
```

# Part IV: CI/CD Pipeline

Due to time constraints, the automated pipeline implementation is outlined as a Design Specification below. This architecture ensures code quality and automated delivery.

1. CI Pipeline (GitHub Actions):
- Triggered on pull requests and pushes to the main branch.
- Steps:
  - Checkout code.
  - Set up Python environment.
  - Install dependencies.
  - Run unit tests (make api-test).
- On success, build and push Docker image to GCR.

2. CD Pipeline (GitHub Actions):
- Triggered on successful CI pipeline completion.
- Steps:
  - Deploy Docker image to Cloud Run using gcloud CLI.
  - Verify deployment by sending a test request to the /health endpoint.
  - Verify deployment by sending a test request to the /predict endpoint with a sample image.