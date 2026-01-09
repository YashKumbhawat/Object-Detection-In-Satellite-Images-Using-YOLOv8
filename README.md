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


### 2️⃣ Convert `.geojson` → YOLO `.txt` Labels
- Each annotation in `xView_train.geojson` is parsed  
- Bounding boxes are converted from: ``` (x_min, y_min, x_max, y_max) ```
- to YOLO format: ``` (class_id, x_center, y_center, width, height) ```
- All values are normalized to the range **[0, 1]**
- A class-mapping list converts xView `type_id`s into YOLO class IDs (0–59)
- One `.txt` label file is created per image  


## 📁 Step 2: YOLO Directory Structure

The following directory structure is created automatically to meet YOLOv8 requirements:

```
images/
├── train/
└── val/

labels/
├── train/
└── val/
```


## 🔀 Step 3: Train / Validation Split (90 / 10)

- All images are randomly shuffled  
- Dataset split:
  - **90%** for training  
  - **10%** for validation  
- Images and corresponding labels are copied into:
  - `images/train`, `images/val`
  - `labels/train`, `labels/val`


## ⚙️ Step 4: Create `data.yaml`

A YOLOv8 configuration file is created containing:
- Dataset base path  
- Training and validation directory paths  
- Number of classes (`nc = 60`)  
- Complete list of xView class names  


## 🧠 Step 5: Train YOLOv8m

YOLOv8m is trained using the following configuration:

```python
model = YOLO("yolov8m.pt")
model.train(
    data="data.yaml",
    epochs=175,
    imgsz=640,
    batch=4,
    workers=2
)
```


## 🧪 Step 6: Select Validation Images for Testing

- 150 images are randomly selected from the validation set
- Selected images are copied into: ```/content/temp_test_150```
- These images are used for:
  - Baseline YOLO inference
  - Super-resolution inference
  - Side-by-side qualitative visualization
  - Detection comparison



## 📊 Step 7: Baseline YOLO Inference (No Super-Resolution)

- YOLOv8 inference is performed on the original validation images
- No super-resolution is applied in this step
- Inference is executed using: ```model.predict(source="temp_test_150")```
- Detection results are saved in: ```results_no_SR/predictions```
- This serves as the baseline performance for comparison



## 🔍 Step 8: Super-Resolution (Bicubic ×2) + YOLO Inference

- Validation images are upscaled by a factor of 2× using bicubic interpolation
- Bicubic interpolation is applied using: ```cv2.INTER_CUBIC```
- Super-resolved images are stored in: ```temp_test_150_SR```
- YOLOv8 inference is performed again on the SR images
- Detection results are saved in: ```results_SR/predictions```

---

## 📊 Key Observations

- YOLOv8 performs well on large and medium-sized objects in high-resolution satellite imagery
- Small and densely packed objects benefit more from higher image resolution
- Bicubic super-resolution improves visual clarity, especially for fine details
- Detection improvements with super-resolution are class-dependent
- Super-resolution increases inference time due to larger image sizes


---

## 📈 Results Summary

- Baseline YOLOv8 inference provides strong overall detection performance
- Super-resolution–based inference shows:
- Improved bounding box confidence for some small objects
- Better visual separation in cluttered regions
- Qualitative comparisons indicate clearer object boundaries in SR images

---

## 🔮 Future Work

- Integrate learning-based super-resolution models (e.g., ESRGAN, Real-ESRGAN)
- Perform quantitative evaluation using mAP@0.5 and mAP@0.5:0.95
- Extend experiments to the full validation and test datasets
- Explore multi-scale training and inference strategies
- Optimize inference speed for real-time deployment

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib
- xView Dataset

---


## 👤 Author

**Yash Kumbhawat**  
Department of Information Technology,  
NITK  

---

## 📜 License

This project is intended for **academic and educational purposes only**.  
You are free to use and modify the code with proper attribution.

