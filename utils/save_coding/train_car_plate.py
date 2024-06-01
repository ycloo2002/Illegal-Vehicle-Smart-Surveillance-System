from ultralytics import YOLO
path = "F:/FYP_Programe/test2/utils/"


import multiprocessing

def main():
    # Load a model
    model = YOLO(path+'car_plate.pt')  # load a pretrained model (recommended for training)


    # Train the model
    results = model.train(data="F:/FYP_Programe/test2/utils/save_coding/brand_dataset/data.yaml", epochs=200, imgsz=640)
    
    pass

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Freeze support for Windows
    main()