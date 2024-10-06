# test_api.py

import requests
import yaml
import cv2
import numpy as np
import os

# Load configuration from config.yaml
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        print("Configuration loaded successfully")
except FileNotFoundError:
    print("Configuration file 'config.yaml' not found.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing configuration file: {e}")
    exit(1)

# Extract configuration parameters
API_URL = config.get('api_url', 'http://localhost:8000/predict')
IMAGE_PATH = config.get('test_image_path', 'examples/test_image.jpg')
VISUALIZE = config.get('visualize', False)

def test_predict():
    with open(IMAGE_PATH, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        print('Success!')
        data = response.json()
        print('Response:')
        print(data)

        if VISUALIZE:
            visualize_results(IMAGE_PATH, data)
    else:
        print(f'Failed with status code {response.status_code}')
        print('Response:')
        print(response.text)

def visualize_results(image_path, data):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image for visualization.")
        return

    boxes = data.get('boxes', [])
    class_ids = data.get('class_ids', [])
    confidences = data.get('confidences', [])

    # Draw bounding boxes on the image
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        xmin, ymin, xmax, ymax = box
        label = f"ID: {class_id}, Conf: {confidence:.2f}"
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image in a window
    cv2.imshow('Detection Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the image to a file
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'result.jpg')
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':
    test_predict()
