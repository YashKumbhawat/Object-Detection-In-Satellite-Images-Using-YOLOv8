# Object Detection on xView Dataset using YOLOv8 with Super-Resolution

## 📌 Project Overview
This project performs **object detection on high-resolution satellite images** using the **xView dataset** and **YOLOv8**.  
The xView dataset is one of the largest publicly available overhead imagery datasets, containing **over 1 million bounding boxes across 60 object classes**, including aircraft, vehicles, ships, buildings, cranes, and more.

In addition to baseline detection, this project investigates whether **Super-Resolution (bicubic ×2 upscaling)** can improve detection performance on small and dense objects.

---

## 🎯 Key Objectives
- Convert xView `.tif` images and `.geojson` annotations into YOLO-compatible format  
- Train **YOLOv8m** on the processed xView dataset  
- Apply **Super-Resolution (bicubic ×2)** on validation images  
- Compare baseline YOLO detections with SR-enhanced detections  

---

## 📂 Dataset Description
- **Dataset Name**: xView Dataset  
- **Source**: Ultralytics Documentation  
- **Link**: https://docs.ultralytics.com/datasets/detect/xview/#dataset-yaml  
- **Type**: Satellite Image Dataset  
- **Number of Classes**: 60  
- **Annotations**: Bounding boxes in GeoJSON format  

---

## 🏗️ Methodology
The project follows a structured pipeline consisting of **dataset preprocessing**, **YOLOv8 training**, and **Super-Resolution–based evaluation**.

---

## 🔄 Step 1: Dataset Preprocessing

The original xView dataset includes:
- Images in `.tif` format  
- Labels stored in a single `.geojson` file  

YOLOv8 requires:
- Images in `.jpg` format  
- Labels in `.txt` format (one file per image)  

### 1️⃣ Convert `.tif` → `.jpg`
All `.tif` images are converted to `.jpg` format and stored in new directories such as:
- `train_images_jpg`
- `val_images_jpg`

---

### 2️⃣ Convert `.geojson` → YOLO `.txt` Labels
- Each annotation in `xView_train.geojson` is parsed  
- Bounding boxes are converted from:
