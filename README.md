# Cancer Features Extraction System

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-Web%20App-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Computational Feature Extraction for Tumor Image Analysis using Handcrafted Methods**

A powerful web-based application designed to analyze medical images, specifically for tumor detection and feature extraction. This tool leverages advanced image processing techniques to identify tumor regions and compute quantitative metrics essential for medical analysis.

---

## ✨ Key Features

- **🖼️ Advanced Image Preprocessing**
  - Grayscale conversion, Gaussian blurring, and intensity rescaling for optimal analysis.

- **🎯 Precise Tumor Segmentation**
  - Utilizes Otsu's thresholding and morphological operations to accurately isolate tumor regions.

- **📊 Comprehensive Feature Extraction**
  - **Shape**: Area, Perimeter, Circularity, Solidity, Eccentricity.
  - **Color**: Mean & Std Dev of RGB channels, Brightness analysis.
  - **Texture (GLCM)**: Contrast, Energy, Homogeneity, Correlation.
  - **Texture (LBP & HOG)**: Local Binary Patterns and Histogram of Oriented Gradients.

- **💾 Interactive & Exportable**
  - Drag-and-drop support, real-time visualization, and CSV export for extraction data.

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python, Flask |
| **Processing** | OpenCV, scikit-image, NumPy, SciPy |
| **Data** | Pandas |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MahmoudAlkhwalda/Computational-Feature-Extraction-for-Tumor-Image-Analysis-using-Handcrafted-Methods.git
   cd Computational-Feature-Extraction-for-Tumor-Image-Analysis-using-Handcrafted-Methods
   ```

2. **Set Up Virtual Environment** (Recommended)
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```
   Open your browser and visit: `http://localhost:5000`

---

## 📂 Project Structure

```text
├── app.py                # Main Flask application
├── cancer_features.py    # Core image processing logic
├── requirements.txt      # Project dependencies
├── static/               # CSS, JS, and images
├── templates/            # HTML templates
└── uploads/              # Temporary storage for uploads
```

---

## 👤 Author

**Mahmoud Alkhwalda**

---

## Acknowledgements

Computer Graphics Project - 5th Semester CS.
