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
IMAGE_PATHS = config.get('test_image_paths', ['examples/test_image.jpg'])
VISUALIZE = config.get('visualize', False)

def test_predict():
    files = []
    valid_image_paths = []
    for image_path in IMAGE_PATHS:
        if not os.path.exists(image_path):
            print(f"Image file '{image_path}' not found.")
            continue
        valid_image_paths.append(image_path)
        files.append(('files', open(image_path, 'rb')))

    if not files:
        print("No valid image files to process.")
        return

    response = requests.post(API_URL, files=files)

    # Close the files to release resources
    for _, file in files:
        file.close()

    if response.status_code == 200:
        print('Success!')
        data = response.json()
        print('Response:')
        print(data)

        if VISUALIZE:
            for image_path, result in zip(valid_image_paths, data):
                visualize_results(image_path, result)
    else:
        print(f'Failed with status code {response.status_code}')
        print('Response:')
        print(response.text)

def visualize_results(image_path, data):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image '{image_path}' for visualization.")
        return

    boxes = data.get('boxes', [])
    class_ids = data.get('class_ids', [])
    confidences = data.get('confidences', [])

    # Draw bounding boxes on the image
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        xmin, ymin, xmax, ymax = box
        label = f"ID: {class_id}, Conf: {confidence:.2f}"
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Optionally, save the image to a file
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")

    # Optionally display the image
    cv2.imshow(f'Detection Results - {output_filename}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_predict()
