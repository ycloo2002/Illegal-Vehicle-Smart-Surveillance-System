from ultralytics import YOLO
import os
import multiprocessing

def main():
    # Load a model
    model = YOLO("F:/fyp_system/yolov8n.pt")  # load a pretrained model (recommended for training)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Train the model
    results = model.train(data="F:/fyp_system/dataset/brand_dataset/data.yaml", epochs=200)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Freeze support for Windows
    main()