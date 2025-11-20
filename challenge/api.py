import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List

import fastapi
from fastapi import File, UploadFile, HTTPException
from PIL import Image
from ultralytics import YOLO

# Logging Configuration 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Object Detector Class 
class ObjectDetector:
    def __init__(self, model_path: str, classes_path: str = None):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.class_names = None
        self.load_resources()

    def load_resources(self):
        """
        Loads the YOLO model (ONNX) and the class mapping JSON.
        """
        try:
            # 1. Load the ONNX Model
            # Ultralytics handles ONNX loading transparently if onnxruntime is installed
            logger.info(f"Loading ONNX model from {self.model_path}...")
            self.model = YOLO(self.model_path, task="detect")
            logger.info("ONNX Model loaded successfully.")

            # 2. Load Class Names from JSON (robust fallback)
            # ONNX exports sometimes lose the internal name mapping.
            if self.classes_path and Path(self.classes_path).exists():
                with open(self.classes_path, 'r') as f:
                    data = json.load(f)

                    names = data.get("names")
                    
                    if isinstance(names, list):
                        # If it's a list like ["forklift", "person"]
                        self.class_names = {i: n for i, n in enumerate(names)}
                    elif isinstance(names, dict):
                        # If it's a dict like {"0": "forklift"}
                        self.class_names = {int(k): v for k, v in names.items()}
                
                logger.info(f"Loaded class names from JSON: {self.class_names}")
            else:
                logger.warning("classes.json not found. Will rely on model internal names.")

        except Exception as e:
            logger.error(f"Critical error loading resources: {e}")
            self.model = None

    def predict(self, image: Image.Image, conf: float = 0.25) -> List[Dict[str, Any]]:
        """
        Runs inference on the image and returns structured results.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        # Run inference
        results = self.model.predict(image, conf=conf)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            # Extract coordinates and scores
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Resolve Class Name
            # Priority: JSON file -> Model Metadata -> Class ID
            if self.class_names:
                class_name = self.class_names.get(class_id, str(class_id))
            elif hasattr(self.model, 'names'):
                 class_name = self.model.names.get(class_id, str(class_id))
            else:
                class_name = str(class_id)

            detections.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })
        return detections

SCRIPT_DIR = Path(__file__).resolve() 
MODEL_BASE_DIR = SCRIPT_DIR.parent / "artifacts"

MODEL_PATH = MODEL_BASE_DIR / "model_best.pt"
CLASSES_PATH = MODEL_BASE_DIR / "classes.json"

# Load model synchronously at module level
# This blocks the process until the model is loaded, bypassing lifespan events.
try:
    GLOBAL_MODEL_DETECTOR = ObjectDetector(str(MODEL_PATH), str(CLASSES_PATH))
    logger.info("Model successfully loaded via synchronous global assignment.")
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to load model globally: {e}")
    GLOBAL_MODEL_DETECTOR = None # Set to None on failure

app = fastapi.FastAPI()

# Endpoints 

@app.get("/health", status_code=200)
async def get_health() -> dict:
    detector = GLOBAL_MODEL_DETECTOR
    
    # Check if detector exists and model loaded successfully
    if detector is None or detector.model is None:
         raise HTTPException(status_code=503, detail="Model not initialized")

    return {
        "status": "model_loaded",
        "model_type": "onnx",
        "model_path": detector.model_path
    }

@app.post("/predict", status_code=200)
async def post_predict(file: UploadFile = File(...)) -> dict:
    detector = GLOBAL_MODEL_DETECTOR
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Received file: {file.filename} with content type: {file.content_type}")
    if not detector or not detector.model:
        raise HTTPException(status_code=501, detail="Model is not available.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        # Run prediction
        detections = detector.predict(image)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    return {
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections
    }