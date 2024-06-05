from ultralytics import YOLO
import matplotlib.pyplot as plt

import multiprocessing

def train():
    # Load a model
    model = YOLO("F:\\fyp_system\\utils\\model\\yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="F:\\FYP_save\\dataset\\Objects365.yaml", epochs=100, imgsz=640)
    
    pass
    
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Freeze support for Windows
    train()