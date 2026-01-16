# Cancer Features Extraction System

A web-based application for analyzing medical images to extract and visualize cancer-related features. This tool processes images to identify tumor regions and computes various quantitative metrics including shape, color, and texture features.

## Features

- **Image Preprocessing**: Convert to grayscale, Gaussian blurring, and intensity rescaling.
- **Tumor Segmentation**: Otsu's thresholding, morphological operations (remove small objects/holes), and overlay visualization.
- **Feature Extraction**:
  - **Shape Features**: Area, perimeter, circularity, solidity, eccentricity, etc.
  - **Color Features**: Mean and standard deviation of RGB channels and brightness.
  - **GLCM Features**: Texture analysis using Gray-Level Co-occurrence Matrix (Contrast, Energy, Homogeneity, etc.).
  - **LBP Features**: Local Binary Patterns for texture classification.
  - **HOG Features**: Histogram of Oriented Gradients.
- **Interactive UI**: Drag-and-drop upload, instant visualization, and CSV export.

## Technologies Used

- **Backend**: Python, Flask
- **Image Processing**: OpenCV, scikit-image, NumPy, SciPy
- **Data Handling**: Pandas
- **Visualization**: Matplotlib
- **Frontend**: HTML5, CSS3, JavaScript

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the application:**
    ```bash
    python app.py
    ```

2.  **Open your browser:**
    Go to `http://localhost:5000`

3.  **Analyze an image:**
    - Click "Choose File" or drag and drop an image (supported: PNG, JPG, JPEG, BMP, GIF).
    - Wait for the processing to complete.
    - View the original image, segmentation mask, and detected region overlay.
    - Review the extracted features table.
    - Click "Download CSV" to save the feature data.

## Project Structure

- `app.py`: Main Flask application handling routes and API endpoints.
- `cancer_features.py`: Core logic for image processing and feature extraction.
- `requirements.txt`: List of Python dependencies.
- `templates/index.html`: Main user interface file.
- `static/`: Contains static assets like CSS.
- `uploads/`: Temporary directory for uploaded files.

## Acknowledgements

Computer Graphics Project - 5th Semester CS.
