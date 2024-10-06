# test_api_unittest.py

import unittest
import requests
import yaml

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

class TestYOLOv9API(unittest.TestCase):
    API_URL = config.get('api_url', 'http://localhost:8000/predict')
    IMAGE_PATH = config.get('test_image_path', 'path_to_your_test_image.jpg')

    def test_predict_endpoint(self):
        with open(self.IMAGE_PATH, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(self.API_URL, files=files)

        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {response.status_code}")
        data = response.json()
        self.assertIn('boxes', data, "Response JSON does not contain 'boxes'")
        self.assertIn('class_ids', data, "Response JSON does not contain 'class_ids'")
        self.assertIn('confidences', data, "Response JSON does not contain 'confidences'")
        self.assertIsInstance(data['boxes'], list, "'boxes' is not a list")
        self.assertIsInstance(data['class_ids'], list, "'class_ids' is not a list")
        self.assertIsInstance(data['confidences'], list, "'confidences' is not a list")

if __name__ == '__main__':
    unittest.main()
