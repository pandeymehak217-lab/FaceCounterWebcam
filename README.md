# 👤 FaceCounterWebcam

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)

A real-time human face detection system using Python, OpenCV, and a Deep Learning-based DNN model (SSD + Caffe).  
Unlike traditional Haar cascades, this model detects only human faces with higher accuracy, reducing false positives.  
The system works with your webcam and displays a live face count.

---

## 📌 Project Overview

This project demonstrates:

- Real-time human face detection using a Deep Learning-based DNN model (SSD + Caffe)  
- Higher accuracy in detecting human faces compared to traditional Haar cascades  
- Live face count display using your webcam  

Ideal for applications requiring real-time face detection and counting.

---

## ⚙️ Requirements

- **Python 3.x**  
- **OpenCV**  
- **NumPy**  

Install dependencies using pip:

```bash
pip install opencv-python numpy
🛠️ Setup & Usage
Clone the repository:
git clone https://github.com/pandeymehak217-lab/FaceCounterWebcam.git
cd FaceCounterWebcam
Download the pre-trained SSD model:
Download the model files from the OpenCV GitHub repository and place them in the project directory.
Run the script:
python face_counter.py
Controls:
Press q to exit the webcam feed.
📂 Repository Structure
FaceCounterWebcam/
├─ face_counter.py                              
├─ deploy.prototxt                              
├─ res10_300x300_ssd_iter_140000_fp16.caffemodel 
├─ README.md                                    
🔗 About
This project showcases the application of Deep Learning-based face detection using OpenCV's DNN module.
By leveraging a pre-trained SSD model with Caffe, it provides accurate and efficient real-time face detection.
