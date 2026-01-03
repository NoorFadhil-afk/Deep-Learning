# 🪖 Helmet Detection using Deep Learning (YOLOv8)

## 📌 Project Overview

This project implements a **deep learning–based helmet detection system** using **YOLOv8**. The model automatically detects whether workers are wearing safety helmets in images or videos, aiming to enhance workplace safety through computer vision.

## 🚀 Features

* Detects helmets in images
* Fast and lightweight YOLOv8n model
* High accuracy for real-time applications
* Simple deployment using Hugging Face Spaces

## 🧠 Model

* **Architecture:** YOLOv8 (You Only Look Once)
* **Variant Used:** YOLOv8n (nano)
* **Type:** One-stage object detection model

## 📂 Dataset

* **Type:** Image dataset
* **Annotations:** YOLO format (bounding boxes)
* **Classes:**

  * Helmet
  * Person (optional)

### Dataset Split

* Training set
* Validation set
* Test set

## ⚙️ Training Details

* **Epochs:** 120
* **Framework:** Ultralytics YOLOv8
* **Loss Functions:**

  * Bounding box loss
  * Classification loss
  * Objectness loss

## 📊 Evaluation Metrics

* Precision
* Recall
* mAP@50
* mAP@50–95

## ✅ Results

The model achieved strong performance in detecting helmets, with improved recall and precision after sufficient training epochs. It performs well in standard lighting and moderately crowded scenes.

## 🌐 Deployment

* Deployed using **Hugging Face Spaces**
* Web interface allows users to upload images and view detection results

## 🛠️ Technologies Used

* Python
* PyTorch
* YOLOv8 (Ultralytics)
* OpenCV
* Hugging Face Spaces

## ⚠️ Challenges

* False detections in crowded scenes
* Performance drop in poor lighting conditions
* Dataset labeling accuracy

## 🔮 Future Improvements

* Add detection for other PPE (vests, gloves)
* Improve dataset diversity
* Support real-time video and CCTV streams

## 👩‍💻 Author

**Noor Ali Fadel**
Bachelor’s degree in Internet E-security

## 📄 License

This project is for educational and research purposes.
