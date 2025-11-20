import os
import json
import yaml
from pathlib import Path
from ultralytics import YOLO

DATA_YAML_PATH = "../data/data.yaml"
ruta_base = Path(DATA_YAML_PATH).parent

assert os.path.exists(DATA_YAML_PATH), f"data.yaml not found in {DATA_YAML_PATH}"

with open(DATA_YAML_PATH, "r") as f:
    data_cfg = yaml.safe_load(f)

class_names = data_cfg.get("names", [])
nc = int(data_cfg.get("nc", len(class_names)))

IMGSZ  = 640

export_dir = Path("artifacts")
export_dir.mkdir(parents=True, exist_ok=True)

best_ckpt = None
for p in Path("runs/detect").rglob("weights/best.pt"):
    best_ckpt = p

model = YOLO(best_ckpt)

if best_ckpt and best_ckpt.exists():
    target = export_dir / "model_best.pt"
    target.write_bytes(best_ckpt.read_bytes())
    print("Wheight export to:", target)
else:
    print("'best.pt' not found")

with open(export_dir / "classes.json", "w") as f:
    json.dump({"nc": int(nc), "names": class_names}, f, indent=2)

try:
    _ = model.export(format="onnx", imgsz=IMGSZ, simplify=False)
    onnx_file = None
    for p in Path(".").rglob("*.onnx"):
        onnx_file = p
        break
    if onnx_file:
        (export_dir / "model.onnx").write_bytes(onnx_file.read_bytes())
        print("ONNX export to:", export_dir / "model.onnx")
    else:
        print("ONNX export failed: file not found after export")
except Exception as e:
    print("Export ONNX not available:", e)

