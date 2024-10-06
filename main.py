# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np
import cv2
import sys
import os
import yaml
import logging

# Adjust sys.path if yolov9 is not in your PYTHONPATH
sys.path.append('./yolov9')

# Import YOLO utilities (adjust if using a different YOLO version)
try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
except ImportError as e:
    logging.error(f"Error importing YOLOv9 modules: {e}")
    exit(1)

class YOLOv9Model:
    def __init__(self, config: dict):
        device_option = config.get('device', 'auto')
        if device_option == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_option)
        self.img_size = config['img_size']
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.classes = config.get('classes', None)
        self.agnostic_nms = config.get('agnostic_nms', False)
        self.weights_path = config['yolo_weights_path']
        self.model = self.load_model(self.weights_path)

    def load_model(self, weights_path: str):
        logging.info(f"Loading YOLOv9 model from {weights_path} on device {self.device}")
        model = attempt_load(weights_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        logging.info("Model loaded successfully")
        return model

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        logging.debug("Preprocessing image")
        img = letterbox(image, self.img_size, stride=32, auto=False)[0]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.float()
        img_tensor /= 255.0  # Normalize to [0,1]
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        logging.debug("Image preprocessed successfully")
        return img_tensor

    def predict(self, img_tensor: torch.Tensor):
        logging.debug("Running inference")
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        logging.debug("Inference completed")
        return pred

    def postprocess(self, predictions, img_tensor_shape, original_shape):
        logging.debug("Postprocessing predictions")
        # Apply NMS
        predictions = non_max_suppression(
            predictions,
            self.confidence_threshold,
            self.iou_threshold,
            classes=self.classes,
            agnostic=self.agnostic_nms
        )
        boxes = []
        class_ids = []
        confidences = []
        for det in predictions:  # detections per image
            if len(det):
                # Rescale boxes from model input size to original image size
                det[:, :4] = scale_boxes(img_tensor_shape[2:], det[:, :4], original_shape)
                for *xyxy, conf, cls in reversed(det):
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    boxes.append([xmin, ymin, xmax, ymax])
                    class_ids.append(int(cls))
                    confidences.append(float(conf))
        logging.debug("Postprocessing completed")
        return boxes, class_ids, confidences

# Initialize logging
logging.basicConfig(
    level=logging.INFO,  # Set to logging.DEBUG for more verbose output
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Load configuration from config.yaml
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully")
except FileNotFoundError:
    logging.error("Configuration file 'config.yaml' not found.")
    exit(1)
except yaml.YAMLError as e:
    logging.error(f"Error parsing configuration file: {e}")
    exit(1)

# Initialize the FastAPI app
app = FastAPI()

# Instantiate the YOLOv9 model with the config
try:
    yolo_model = YOLOv9Model(config=config)
except Exception as e:
    logging.error(f"Error initializing YOLOv9Model: {e}")
    exit(1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image data
        image_data = await file.read()
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logging.error("Invalid image data")
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img_rgb.shape[:2]  # Height, Width

        # Preprocess image
        img_tensor = yolo_model.preprocess_image(img_rgb)

        # Inference
        predictions = yolo_model.predict(img_tensor)

        # Postprocess
        boxes, class_ids, confidences = yolo_model.postprocess(predictions, img_tensor.shape, original_shape)

        # Return the results as JSON
        results = {'boxes': boxes, 'class_ids': class_ids, 'confidences': confidences}
        return JSONResponse(content=results)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Start the application with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
