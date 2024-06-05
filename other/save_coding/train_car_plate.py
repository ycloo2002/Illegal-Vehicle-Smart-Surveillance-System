from ultralytics import YOLO
path = "F:/FYP_Programe/test2/utils/"
import matplotlib.pyplot as plt

import multiprocessing

def train():
    # Load a model
    model = YOLO("F:\\fyp_system\\utils\\model\\yolov8s.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="F:\\FYP_save\\dataset\\car_plate_3\\data.yaml", epochs=100, imgsz=640)
    print(results)


def val():
    model = YOLO("F:\\fyp_system\\utils\\model\\car_plate_v3.pt")
    results = model.val(data="F:\\FYP_save\\dataset\\car_plate_3\\data.yaml", imgsz=640)

    # Print validation results
    print(results)

    # Visualize some of the validation metrics
    plt.plot(results.history['mAP'], label='mAP')
    plt.plot(results.history['precision'], label='Precision')
    plt.plot(results.history['recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Validation Metrics')
    plt.show()
    
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Freeze support for Windows
    train()