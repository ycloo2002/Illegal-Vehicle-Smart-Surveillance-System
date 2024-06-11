import cv2
import os
import yaml
import numpy as np
import shutil 
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, 
    HueSaturationValue, Rotate, Resize
)
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

if __name__ == '__main__':
    # Define paths
    dataset_yaml = "F:/FYP_save/dataset/brand_v2/data.yaml"
    preprocessed_output_path = 'preprocessed_dataset'


    # Load a YOLOv8 model, pretrained on COCO
    model = YOLO('yolov8m.pt')  # or 'yolov8m.pt', 'yolov8l.pt', etc. for different sizes

    # Train the model
    model.train(data=dataset_yaml, epochs=50, batch=16, imgsz=640, 
                workers=4, device=0, project='runs/train', name='exp', 
                pretrained=True, optimizer='Adam', lr0=0.01, weight_decay=0.0005, 
                momentum=0.937, patience=5, verbose=True)

    # Evaluate the model
    metrics = model.val(data=dataset_yaml)
    print(metrics)
