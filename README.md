# Real-Time Face Mask Detection System 

This project implements a real-time face mask detector using an active webcam feed. It leverages Transfer Learning with MobileNetV2 -an industry standard architecture for lightweight, real-time edge vision applications—to classify whether a person in the frame is wearing a mask or not.

#  Project Overview

* Objective: Build a deployable computer vision system to detect face masks in real-time.
* Architecture: MobileNetV2 (Pre-trained on ImageNet) with a custom Dense head for binary classification (Mask vs. No Mask).
* Dataset: Kaggle Face Mask Dataset (Images & XML bounding box annotations).
* Environment: Python, Jupyter/Google Colab (for training), PyCharm (for local deployment).

# Tech Stack & Libraries

* Deep Learning: TensorFlow / Keras
* Computer Vision: OpenCV ('cv2')
* Data Processing: NumPy, Pandas, BeautifulSoup (for XML parsing)
* Visualization: Matplotlib, Seaborn

# Key Files

* mask_detector.h5: The fine-tuned MobileNetV2 model weights.
* webcam_dectector.py: The deployment script that captures local webcam footage, extracts face regions of interest (ROIs), and passes them to the model for real-time inference.

# How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/divyanisinha2005-creator/-Real-Time-Face-Mask-Detector---Divyani-sinha.git](https://github.com/divyanisinha2005-creator/-Real-Time-Face-Mask-Detector---Divyani-sinha.git)
