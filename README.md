# 🧬 YOLOv8-Based Esophageal Cancer Detection

## 📌 Overview
This project focuses on detecting cancerous regions in endoscopy images using a YOLOv8 deep learning model.  
A user-friendly **Streamlit web application** allows users to upload endoscopy images and visualize predictions.

The model was trained on annotated medical images and is designed to make **conservative predictions**, prioritizing accuracy and reducing false positives.

---

## Features
- Endoscopy image-based cancer detection
- YOLOv8 object detection model
- Streamlit-based interactive web interface
- Supports image upload and real-time prediction
- Handles cases with **no detection** when confidence is low

---

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- Streamlit
- OpenCV
- PyTorch
- Roboflow (for data annotation)
- Google Colab (for model training)


---

##  Project Structure
yolov8-esophageal-neoplasia-detection/
│
├── src/
│   ├── app.py        # Streamlit application for inference
│   └── best.pt       # Trained YOLOv8 model weights
│
├── data/
│   ├── esophagus/    # Annotated cancerous endoscopy images
│   └── non_esophagus/ # Annotated non-cancerous images
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 🧪 Dataset
- Endoscopy images annotated using **Roboflow**
- Two classes:
  - Cancerous (Esophagus Neoplasia)
  - Non-Cancerous
- Dataset prepared for YOLOv8 format

---
## 🏋️ Model Training
The YOLOv8 model was trained on annotated endoscopy images using **Google Colab** to utilize GPU acceleration.
After training, the best-performing model weights were exported as `best.pt` and used for inference in the Streamlit application.

---
## 🧠 Model Weights (`best.pt`)
The `best.pt` file contains the trained YOLOv8 model parameters and is loaded by the application to perform predictions on new endoscopy images.
---


## 📊 Model Performance
- Achieved approximately **84% accuracy**
- Designed to avoid incorrect predictions by returning **no detection** if confidence is low
- Performance depends on image quality and lighting conditions

---
## ⚙️ How to Run
1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies using `requirements.txt`  
4. Run the Streamlit app
---

## ⚙️ Setup Instructions (Windows)

###1️.Clone the Repository
```bat
git clone <your-repository-url>
cd yolov8-esophageal-neoplasia-detection
2️.Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

3️.Install Dependencies
pip install -r requirements.txt

4️.Run the Application
streamlit run src/app.py


The app will open in your browser automatically.
---
## Results

Upload an endoscopy image

The model highlights detected cancerous regions

If confidence is low, no detection is shown (expected behavior)

Limitations

Model may not detect cancer in low-quality images

Limited dataset size

Not intended for clinical diagnosis (research prototype only)

 Future Improvements

Increase dataset size

Improve data augmentation

Add confidence score display

Deploy using Docker or cloud platforms

Clinical expert validation

Author

Tirumanyam Vamsi Krishna Reddy
B.Tech – Artificial Intelligence & Data Science