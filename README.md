# 📡 CSI-Sense: Wi-Fi CSI-Based Human Activity Recognition Using Deep Learning

CSI-Sense is a **contactless human activity recognition system** that leverages **Wi-Fi Channel State Information (CSI)** captured using ESP32 devices and a **Convolutional Neural Network (CNN)** to classify human activities.

The system performs human activity recognition **without cameras or wearable sensors**, making it **privacy-preserving** and suitable for smart indoor environments.

---

## 🎯 Project Objectives
- Capture fine-grained Wi-Fi CSI data using ESP32 devices
- Analyze CSI variations caused by human motion
- Train a deep learning model to recognize human activities
- Perform real-time inference using a Python-based system

---

## 🔧 System Overview
- **Hardware:** ESP32 (Transmitter) + ESP32 (Receiver)
- **CSI Collection:** ESP-IDF CSI Tool
- **Activities Recognized:** Sitting, Standing, Walking
- **AI Model:** Convolutional Neural Network (CNN)
- **Inference Platform:** Laptop (Python)

---

## 🏗 System Architecture
The proposed system captures Wi-Fi CSI data between two ESP32 devices. Human movement alters the wireless propagation environment, which is reflected in CSI amplitude variations. These variations are processed and fed into a CNN model to classify activities.

<img width="1536" height="1024" alt="ChatGPT Image Dec 24, 2025, 02_17_25 AM" src="https://github.com/user-attachments/assets/7d93acf0-7442-4a1f-9aff-aba60ca4338e" />


---

## 📡 CSI Data Acquisition
Wi-Fi Channel State Information (CSI) is collected using two ESP32 devices configured as transmitter and receiver. The **ESP-IDF CSI tool** is used to extract subcarrier-level CSI measurements.

Different human activities introduce distinct temporal patterns in CSI amplitude due to signal reflection, absorption, and scattering caused by body movement.

---

## 📂 Dataset Description
- CSI data stored in CSV format
- Separate recordings for each activity class
- Multiple time windows per activity
- Data preprocessing includes:
  - Noise filtering
  - Normalization
  - Temporal segmentation
  - Reshaping for CNN input

⚠️ **The full CSI dataset is not included in this repository due to size limitations.**

---

## 🧠 Deep Learning Model
A **Convolutional Neural Network (CNN)** is employed to learn discriminative spatial-temporal features from CSI sequences.

### Model Characteristics
- 1D convolution layers for CSI sequence learning
- Pooling layers for feature abstraction
- Fully connected layers for classification
- Softmax output layer for activity prediction

---

## 📈 Model Evaluation

### Confusion Matrix
The confusion matrix is used to evaluate the classification performance across different human activities.

<img width="634" height="511" alt="image" src="https://github.com/user-attachments/assets/863ac7cb-a7a2-4917-b1d7-fe3964255c73" />


---

## 📊 Results Summary
- **Training Accuracy:** ~92–95%
- **Validation Accuracy:** ~88–91%
- **Activities Classified:**
  - Sitting
  - Standing
  - Walking

The results demonstrate that Wi-Fi CSI can be effectively utilized for **contactless human activity recognition** using deep learning techniques.

---

## 🧪 Tools & Technologies
- ESP32
- ESP-IDF
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Flask
- Matplotlib

---

## 📦 Model Availability
Due to GitHub file size limitations, the trained CNN model (`.h5`) is not included in this repository.

The repository provides:
- Complete CSI preprocessing pipeline
- Model architecture and training code
- Inference and evaluation scripts

Pre-trained model files can be generated using the training scripts or shared upon request.

---

## 🚀 Future Enhancements
- Add more activities (running, falling, gestures)
- Real-time visualization dashboard
- On-device inference using TinyML
- Multi-person activity recognition
- Cloud-based deployment

---

## 👨‍💻 Author
**Kasun Wickramasingha**  
Final Year Undergraduate – Electrical & Electronic Engineering  
Machine Learning | Embedded Systems | Wi-Fi Sensing

---

## ⭐ Acknowledgment
This project was developed as a **final year research project** exploring AI-driven wireless sensing for privacy-preserving human activity recognition.
