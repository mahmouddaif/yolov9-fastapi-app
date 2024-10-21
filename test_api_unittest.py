# test_api_unittest.py

import unittest
import requests
import yaml
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

class TestYOLOv9API(unittest.TestCase):
    API_URL = config.get('api_url', 'http://localhost:8000/predict')
    IMAGE_PATHS = config.get('test_image_paths', ['path_to_your_test_image1.jpg', 'path_to_your_test_image2.jpg'])

    def test_predict_endpoint(self):
        files = []
        for image_path in self.IMAGE_PATHS:
            if not os.path.exists(image_path):
                self.fail(f"Image file '{image_path}' not found.")
            files.append(('files', open(image_path, 'rb')))

        response = requests.post(self.API_URL, files=files)

        # Close the files to release resources
        for _, file in files:
            file.close()

        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {response.status_code}")

        data = response.json()

        # Assert that the response is a list
        self.assertIsInstance(data, list, "Response is not a list")

        # Assert that the number of results matches the number of images sent
        self.assertEqual(len(data), len(self.IMAGE_PATHS), "Number of results does not match number of images sent")

        # For each result, perform assertions
        for idx, result in enumerate(data):
            self.assertIn('boxes', result, f"Result for image {idx} does not contain 'boxes'")
            self.assertIn('class_ids', result, f"Result for image {idx} does not contain 'class_ids'")
            self.assertIn('confidences', result, f"Result for image {idx} does not contain 'confidences'")
            self.assertIsInstance(result['boxes'], list, f"'boxes' for image {idx} is not a list")
            self.assertIsInstance(result['class_ids'], list, f"'class_ids' for image {idx} is not a list")
            self.assertIsInstance(result['confidences'], list, f"'confidences' for image {idx} is not a list")

if __name__ == '__main__':
    unittest.main()
