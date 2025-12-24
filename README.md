DATASET LINK : https://docs.ultralytics.com/datasets/detect/xview/#dataset-yaml

PROJECT OVERVIEW:

This project performs object detection on high-resolution satellite images using the xView dataset and YOLOv8. The xView dataset contains over 1 million bounding boxes across 60 classes such as aircraft, vehicles, ships, buildings, cranes, etc.

Key Objectives:

Convert xView .tif images and .geojson labels into YOLO-compatible format

Train YOLOv8m on the processed dataset

Apply Super-Resolution (bicubic x2) to test if upscaling improves detection

Compare baseline YOLO detections with SR-enhanced detections


(1) PREPROCESSING THE DATASET


The original xView dataset includes:

Images in .tif format

Labels in a single .geojson file

YOLO requires:

Images in .jpg format

Labels in .txt format (one file per image)

Step 1: Convert .tif → .jpg
All .tif images are converted to .jpg and stored in new folders such as train_images_jpg and val_images_jpg.

Step 2: Convert .geojson → YOLO .txt labels
Each annotation in xView_train.geojson is read and converted from (x_min, y_min, x_max, y_max) into YOLO format:
(class_id, x_center, y_center, width, height) normalized to the range 0–1.
A class mapping list converts xView type_ids into YOLO class IDs (0–59).
One .txt file is created per image.


(2) CREATE YOLO DIRECTORY STRUCTURE


The following folders are created automatically:

images/train
images/val
labels/train
labels/val

This structure is required by YOLOv8.


(3) TRAIN / VALIDATION SPLIT (90/10)


All images are randomly shuffled, then split into:

90% for training

10% for validation

Both images and corresponding labels are copied into:
images/train, images/val
labels/train, labels/val


(4) CREATE data.yaml FOR YOLOv8


This configuration file includes:

Dataset base path

Train and validation directory paths

Number of classes (nc = 60)

Full list of xView class names


(5) TRAIN YOLOv8m

YOLOv8m is trained using:

model = YOLO("yolov8m.pt")
model.train(
data="data.yaml",
epochs=175,
imgsz=640,
batch=4,
workers=2
)

The best weights file is saved at:
runs/detect/train/weights/best.pt


(6) SELECT 150 VALIDATION IMAGES FOR TESTING

150 random images from the validation set are selected and copied into:
/content/temp_test_150

These images are used for:

Baseline evaluation

Super-resolution evaluation

Side-by-side visualization

Metric comparison


(7) BASELINE YOLO INFERENCE (NO SUPER-RESOLUTION)

YOLOv8 is run on the 150 original images with:

model.predict(source="temp_test_150")

The annotated detection results are saved in:
results_no_SR/predictions


(8) SUPER-RESOLUTION (BICUBIC x2) + YOLO

Since no external SR model is loaded, bicubic interpolation is used:

img_sr = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_CUBIC)

The upscaled images are saved in:
temp_test_150_SR

YOLO inference is run again on SR images and the results are saved in:
results_SR/predictions


(9) VISUALIZE RAW VS SR IMAGES (NO DETECTIONS)

A function displays the raw image and its SR version side-by-side to visually inspect improvements in clarity.





