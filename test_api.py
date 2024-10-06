# test_api.py

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

# Extract the API URL and image path from the config
API_URL = config.get('api_url', 'http://localhost:8000/predict')
IMAGE_PATH = config.get('test_image_path', 'path_to_your_test_image.jpg')

def test_predict():
    with open(IMAGE_PATH, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        print('Success!')
        print('Response:')
        print(response.json())
    else:
        print(f'Failed with status code {response.status_code}')
        print('Response:')
        print(response.text)

if __name__ == '__main__':
    test_predict()
