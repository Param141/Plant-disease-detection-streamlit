# Plant-disease-detection-streamlit

# 🍆 AI Plant Disease Detection System

## Overview
This repository contains the code and deployment files for an AI-powered Plant Disease Detection System. Currently, the system focuses on diagnosing diseases in Brinjal (Eggplant) leaves using a deep learning model.

**Live App:** [https://plant-disease-detection-app-kwqyy5xxdkv3kva3mpuug6.streamlit.app/]

## Features
* **High Accuracy Image Classification:** Uses a fine-tuned ResNet50V2 Convolutional Neural Network (CNN).
* **Real-time Inference:** Upload an image of a leaf and receive instant predictions.
* **Supported Brinjal Classes:**
  * Healthy Leaves
  * Little Leaf Disease
  * Phomopsis Blight

## Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, Pillow, NumPy
* **Deployment:** Streamlit Community Cloud

## How to Run Locally
1. Clone the repository: `git clone https://github.com/ParamSharma123/brinjal-disease-detector.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Future Scope
* Integrate additional plant models (e.g., Ridge Gourd).
* Expand the dataset for higher generalization.
