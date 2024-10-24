# YOLOv9 FastAPI Inference Application

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up the Environment](#2-set-up-the-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
  - [Running with Python](#running-with-python)
  - [Running with Docker](#running-with-docker)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
- [Inference Device Selection (CPU/GPU)](#inference-device-selection-cpugpu)
- [Testing the API](#testing-the-api)
  - [Visualization of Detection Results](#visualization-of-detection-results)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Additional Resources](#additional-resources)
- [Contact](#contact)

---

## Introduction

This project provides a FastAPI application that wraps a YOLOv9 model for object detection inference. It allows users to perform object detection on images via a RESTful API. The application supports both CPU and GPU inference and can be easily deployed using Docker.

**New Feature:** The API now supports batch inference, allowing multiple images to be processed in a single request.

---

## Features

- **Object Detection API**: Exposes a `/predict` endpoint for image inference.
- **Batch Inference Support**: Allows multiple images to be processed in a single request.
- **Flexible Device Usage**: Supports inference on both CPU and GPU.
- **Configuration Management**: Uses a `config.yaml` file for easy configuration.
- **Dockerized Deployment**: Provides Dockerfiles for containerization.
- **Logging**: Includes logging for monitoring and debugging.
- **Error Handling**: Handles exceptions gracefully and provides meaningful error messages.
- **Swagger UI**: Automatically generated API documentation available at `/docs`.
- **Customizable**: Easy to extend and modify for different use cases.
- **Automated Testing**: Includes test scripts to verify the API functionality.
- **Visualization Option**: Ability to visualize detection results in test scripts, controlled via configuration.

---

## Prerequisites

- **Python 3.8+**
- **Git**
- **CUDA-compatible GPU** (optional, for GPU inference)
- **Docker** (optional, for containerized deployment)
- **NVIDIA Docker Toolkit** (optional, for GPU support in Docker)
- **PyTorch** (ensure compatibility with your CUDA version if using GPU)
- **OpenCV** (for visualization, included in `requirements.txt`)

---

## Project Structure

```plaintext
yolov9-fastapi-app/
├── main.py                # FastAPI application code
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── Dockerfile             # Dockerfile for building the image
├── weights/
│   └── best.pt            # YOLOv9 weights file
├── yolov9/                # Cloned YOLOv9 repository
├── examples/              # Directory for example images
├── output/                # Directory for output images (visualizations)
├── test_api.py            # Test script for the API with visualization option
├── test_api_unittest.py   # Unit test script for the API
└── README.md              # Project documentation (this file)
```

---

## Installation

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/mahmouddaif/yolov9-fastapi-app.git
cd yolov9-fastapi-app
```

### 2. Set Up the Environment

It's recommended to use a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

- On **Linux/macOS**:

  ```bash
  source venv/bin/activate
  ```

- On **Windows**:

  ```bash
  venv\Scripts\activate
  ```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Ensure that you have installed all the dependencies, including those required by YOLOv9.

---

## Configuration

Edit the `config.yaml` file to configure the application:

```yaml
# config.yaml

yolo_weights_path: 'weights/best.pt'          # Path to your YOLOv9 weights file
device: 'auto'                                # Options: 'auto', 'cuda', 'cpu'
img_size: 640                                 # Image size for inference
confidence_threshold: 0.25                    # Confidence threshold for detections
iou_threshold: 0.45                           # IoU threshold for NMS
classes: null                                 # Filter by class indices, e.g., [0, 1, 2]
agnostic_nms: false                           # Class-agnostic NMS
test_image_paths:                             # Paths to test images for batch inference
  - 'examples/test_image1.jpg'
  - 'examples/test_image2.jpg'
  - 'examples/test_image3.jpg'
api_url: 'http://localhost:8000/predict'      # API endpoint URL
visualize: false                              # Set to true to enable visualization
```

- **`test_image_paths`**: List of image paths for testing batch inference.
- **`visualize`**: Controls whether to display and save visualization of detection results in test scripts. Set to `false` by default.

---

## Running the Application

### Running with Python

Start the FastAPI application using Uvicorn:

```bash
python main.py
```

The application will be available at `http://0.0.0.0:8000`.

### Running with Docker

#### Build the Docker Image

**For CPU Inference:**

```bash
docker build -t yolov9-fastapi:cpu .
```

**For GPU Inference:**

```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime -t yolov9-fastapi:gpu .
```

**Notes:**

- Ensure you have the appropriate CUDA version matching your GPU and PyTorch compatibility.
- You can adjust the `BASE_IMAGE` to the desired PyTorch version and CUDA version.

#### Run the Docker Container

**For CPU Inference:**

```bash
docker run -d -p 8000:8000 yolov9-fastapi:cpu
```

**For GPU Inference:**

Ensure you have the NVIDIA Docker toolkit installed.

```bash
docker run --gpus all -d -p 8000:8000 yolov9-fastapi:gpu
```

---

## Inference Device Selection (CPU/GPU)

- **`device` Option in `config.yaml`:**

  - `'auto'`: Use GPU if available, otherwise fallback to CPU.
  - `'cuda'`: Force GPU usage. Will raise an error if GPU is not available.
  - `'cpu'`: Force CPU usage, even if a GPU is available.

- **Environment Variable (Optional):**

  You can override the device selection using the `DEVICE` environment variable.

  ```bash
  docker run -e DEVICE='cpu' -d -p 8000:8000 yolov9-fastapi:cpu
  ```

- **Example:** Set `device: 'cuda'` in `config.yaml` for GPU inference.

---

## Testing the API

### Test Script (`test_api.py`)

A test script is provided to verify the API functionality. The script reads image paths and the API URL from the `config.yaml` file and can visualize detection results based on the `visualize` configuration option.

**Instructions:**

1. Ensure your FastAPI application is running.
2. Set `visualize` in `config.yaml`:

   - To enable visualization:

     ```yaml
     visualize: true
     ```

   - To disable visualization:

     ```yaml
     visualize: false
     ```

3. Specify test image paths in `config.yaml`:

   ```yaml
   test_image_paths:
     - 'examples/test_image1.jpg'
     - 'examples/test_image2.jpg'
     - 'examples/test_image3.jpg'
   ```

4. Run the test script:

   ```bash
   python test_api.py
   ```

**Visualization Output:**

- When `visualize` is set to `true`, the script will:

  - Display each image with detection results in a window.
  - Save the visualized images to the `output/` directory with their original filenames.

- When `visualize` is set to `false`, the script will only print the API response without displaying or saving images.

**Note:** Ensure the `test_image_paths` and `api_url` in `config.yaml` are correctly set.

### Visualization of Detection Results

- **Enabling Visualization:**
  - Open `config.yaml` and set `visualize: true`.
  - Run `python test_api.py`.
  - The script will display and save the images with detection results.
- **Disabling Visualization:**
  - Set `visualize: false` in `config.yaml`.
  - Run `python test_api.py`.
  - The script will only output the API response.

### Unit Test Script (`test_api_unittest.py`)

A unit test script using the `unittest` framework is also provided. It reads configuration from the `config.yaml` file and tests the batch inference functionality.

**Instructions:**

1. Ensure your FastAPI application is running.
2. Specify test image paths in `config.yaml`:

   ```yaml
   test_image_paths:
     - 'examples/test_image1.jpg'
     - 'examples/test_image2.jpg'
   ```

3. Run the unit test script:

   ```bash
   python -m unittest test_api_unittest.py
   ```

**Note:** The unit test script does not include visualization to keep tests automated and non-interactive.

---

## API Documentation

FastAPI provides interactive API documentation available at `http://localhost:8000/docs`. You can use this interface to test the `/predict` endpoint and view the API schema.

---

## Examples

Below are some examples of how to use the API.

### Example 1: Batch Inference with Multiple Images

```bash
curl -X POST "http://localhost:8000/predict" \
-F "files=@examples/test_image1.jpg" \
-F "files=@examples/test_image2.jpg" \
-F "files=@examples/test_image3.jpg"
```

**Response:**

```json
[
  {
    "boxes": [[34, 50, 200, 300]],
    "class_ids": [0],
    "confidences": [0.85]
  },
  {
    "boxes": [[150, 80, 400, 350]],
    "class_ids": [1],
    "confidences": [0.78]
  },
  {
    "boxes": [],
    "class_ids": [],
    "confidences": []
  }
]
```

### Example 2: Using Postman for Batch Inference

1. Open Postman and create a new `POST` request to `http://localhost:8000/predict`.
2. Under the `Body` tab, select `form-data`.
3. Add multiple keys named `files` of type `File` and select the images you want to test.
4. Send the request and view the response.

### Example 3: Visualizing Results in Test Script

- Enable visualization in `config.yaml`:

  ```yaml
  visualize: true
  ```

- Run the test script:

  ```bash
  python test_api.py
  ```

- The script will display each image with bounding boxes and save them to the `output/` directory.

---

## Troubleshooting

- **Invalid Image Error:**

  Ensure that the images you're sending are valid and in supported formats (e.g., JPEG, PNG). The images should not be corrupted.

- **ModuleNotFoundError:**

  Check that all dependencies are installed and that the `yolov9` repository is correctly cloned. Ensure that `sys.path` is adjusted correctly in `main.py` to include the `yolov9` directory.

- **GPU Not Detected:**

  - Ensure that your system has a CUDA-compatible GPU.
  - Verify that NVIDIA drivers and CUDA are properly installed.
  - If using Docker, make sure the NVIDIA Docker toolkit is installed, and the container is run with `--gpus all`.

- **Permission Denied Errors:**

  Ensure that the weights file and other resources have the appropriate permissions and are accessible by the application. Check file permissions and ownership.

- **Configuration File Not Found:**

  Make sure that `config.yaml` is present in the working directory and correctly formatted. Check for syntax errors in the YAML file.

- **Docker Build Errors:**

  - Ensure that all files required for the build are present.
  - Check for network connectivity issues if the build cannot fetch packages or clone repositories.
  - Verify that the correct base image is specified.

- **Runtime Errors:**

  Check the application logs for detailed error messages. Logs are printed to the console and can help diagnose issues.

- **Visualization Issues:**

  - If the visualization window does not appear, ensure that you are not running the script in a headless environment.
  - For servers without a GUI, consider disabling the display code or running the script locally.

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please ensure that your code adheres to the project's coding standards and passes all tests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **YOLOv9 Repository:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- **YOLOv9 Paper:** [YOLOv9: Scaling Up Yolo for Better Performance](https://arxiv.org/abs/2402.13616)
- **FastAPI Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **PyTorch:** [PyTorch](https://pytorch.org/)
- **Uvicorn:** [Uvicorn](https://www.uvicorn.org/)
- **OpenAI's GPT-4:** For providing assistance in developing this project.

---

## Additional Resources

- **YOLOv9 Paper:** [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)
- **FastAPI Documentation:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Docker Documentation:** [https://docs.docker.com/](https://docs.docker.com/)
- **PyTorch Documentation:** [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

## Contact

If you have any questions or need further assistance, feel free to:

- **Open an Issue:** [https://github.com/mahmouddaif/yolov9-fastapi-app/issues](https://github.com/mahmouddaif/yolov9-fastapi-app/issues)

---

**Note:** Adjust paths, configuration options, and other settings according to your specific environment and requirements. Ensure that all dependencies are correctly installed, and the weights file is in the correct location.

For further assistance, please refer to the official documentation of the respective tools or raise an issue in this repository.
