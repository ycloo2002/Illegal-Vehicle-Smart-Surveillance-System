import time
import cv2
import easyocr
import re
import csv
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import threading
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QColorDialog,
    QGroupBox,
    QStackedWidget,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot


def show_output(frame):
    
    # Naming a window 
    cv2.namedWindow("Car_Plate_Detection", cv2.WINDOW_NORMAL) 
        
    # Using resizeWindow() 
    cv2.resizeWindow("Car_Plate_Detection", 1500, 1000)
    
    #window posistion
    cv2.moveWindow("Car_Plate_Detection", 100, 50) 
    
    #show image 
    cv2.imshow("Car_Plate_Detection", frame)
        

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True   

def drawbox(img,x1,x2,y1,y2,label_text,color = (255, 0, 0),thickness = 5):
    
    start_point = (x1, y1)
    end_point = (x2, y2)
    label_text = str(label_text)

    # Draw the rectangle on the image
    cv2.rectangle(img, start_point, end_point, color, thickness)
    
    # Add the label text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0,0,0) #black
    (label_width, label_height), _ = cv2.getTextSize(label_text, font, font_scale,thickness)
    label_position = (start_point[0], start_point[1] - 5) 
    
    start_point = label_position
    end_point = (label_position[0] + label_width, label_position[1] - label_height)
    cv2.rectangle(img, start_point, end_point, color, thickness=cv2.FILLED)

    cv2.putText(img, label_text, label_position, font, font_scale, font_color, thickness=5)

def data_and_time():
    date = datetime.now()
    return str(date.strftime('%Y-%m-%d %H-%M-%S'))

def check_duplicated_plate_numbers_no(csv_file_path,vehicleplate):
    # Read existing data from CSV file
    no = 0
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for col in reader:
            if col[1] == vehicleplate:
                return False,0
            no+=1      
            
        return True,no   
    
def insert_csv(no,data,csv_path,message):

    data_with_index = [str(no)] + data + [message]
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_with_index)   
    
def create_csv(path):

    csv_file_path = f"{path}/result.csv"
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['No','Plate_number', 'Type','Brand','Colour','Message'])
        print(f"CSV file '{csv_file_path}' successful create.")
        
    return csv_file_path
            
def text_reconise(lp_crop,reader):
    
    #run the text prediction to get the text                         
    text_result = reader.readtext(lp_crop)
     
    #if non of the result showing out    
    if text_result:
                                
        #define 
        lp_text =""
        total_score =0
        
        #combine the text (which some of the lp having two rows)
        for text_detact in text_result:
            bbox, text, score = text_detact

            lp_text += text
            total_score += score
                
        avg_lp_score = total_score/len(text_result)
               
                                
        #continue when the lp reconigse having more that equal to 80%    
        if avg_lp_score >= 0.8:   
            
            #lp pattern set : start from letter and must contain two character and at least one number and not contain any symbol   
            pattern =r'^[a-zA-Z][a-zA-Z0-9]*[0-9][a-zA-Z0-9]*$'
            
            #continue when the pattern is correct.
            if re.match(pattern, lp_text):
                
                #remove the spacing
                lp_text = lp_text.upper().replace(' ', '')
                return [True,lp_text,avg_lp_score]

    return [False,"",""]

def color_reconigse(image, model,device):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image = transform(pil_image).unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device
    image = image.to(device)

    # Set the model to evaluation mode and make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

def insert_table_info(table,data,image_path,invalid=False,message="",vehicle_onwer=""):
    
    if invalid : choice = table.table_warnning 
    else : choice = table.table_info
    
    row_count = choice.rowCount()
    choice.insertRow(row_count)
    
    # Add image to the Photo column
    pixmap = QPixmap(image_path)
    icon = QIcon(pixmap)
    item = QTableWidgetItem()
    item.setIcon(icon)
    item.setTextAlignment(Qt.AlignCenter)
    item.setSizeHint(pixmap.size()) 
    
    choice.setItem(row_count, 0, item)  
    
    
    choice.setItem(row_count, 1, QTableWidgetItem(data[0]))
    choice.setItem(row_count, 2, QTableWidgetItem(data[1]))
    choice.setItem(row_count, 3, QTableWidgetItem(data[2]))
    choice.setItem(row_count, 4, QTableWidgetItem(data[3]))
    
    if invalid:
        choice.setItem(row_count, 5, QTableWidgetItem(message))
        choice.setItem(row_count, 6, QTableWidgetItem(vehicle_onwer))
                                
def check_invalid_vehicle(result_data,path):
    
    #default
    code = "notfound"
    onwer_name = ""
    
    #open the database file
    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for col in reader:
            if col[2] == result_data[1]: # true in vehicle plate 
                code = "r"
                onwer_name = col[1]
                #type
                if col[3] != result_data[2]: code += "T"
                
                #brand
                if col[4] != result_data[3]: code += "B"
                
                #color
                if col[5] != result_data[4]: code += "C"
                
                break
        
    match code:
        case "notfound" : return True,"This License Plate not register in the system.",onwer_name
        
        case "rTBC" : return True,"Invalid Vehicle Type, Brand and Colour for this License Plate.",onwer_name
        case "rTB" : return True,"Invalid Vehicle Type, Brand and Colour for this License Plate.",onwer_name
        case "rTC" : return True,"Invalid Vehicle Type and Colour for this License Plate.",onwer_name
        case "rT" : return True,"Invalid Vehicle Type for this License Plate.",onwer_name
            
        case "rB" : return True,"Invalid Vehicle Brand for this License Plate.",onwer_name
        case "rBC" : return True,"Invalid Vehicle Brand and Colour for this License Plate.",onwer_name
            
        case "rC" : return True,"Invalid Vehicle Colour for this License Plate.",onwer_name
        
        case "r" : return False,"No error found.",onwer_name
  
class Detaction:

    def __init__(load):
    # Load the model
        load.vehicel_model = YOLO('utils/yolov8n.pt')
        load.plate_detection = YOLO('utils/car_plate_v2.pt')
        load.brand_detection = YOLO('utils/brand.pt')
    
        # Load the model for color recorigse
        color_model = models.googlenet(pretrained=False, aux_logits=True)  # Set aux_logits to True to match the saved model
        num_ftrs = color_model.fc.in_features
        color_model.fc = nn.Linear(num_ftrs, 15)  # Adjust num_classes to match your dataset
        color_model.aux1.fc2 = nn.Linear(color_model.aux1.fc2.in_features, 15)
        color_model.aux2.fc2 = nn.Linear(color_model.aux2.fc2.in_features, 15)

        model_path = "utils/colour.pth"
        color_model.load_state_dict(torch.load(model_path))
        load.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        color_model = color_model.to(load.device)
        color_model.eval()  # Set the model to evaluation mode
        
        load.color_model = color_model
        
        # Initialize the OCR reader
        load.reader = easyocr.Reader(['en'], gpu=True)
        
        load.vehicles = {2:"car",5:'bus', 7:'truck'} # 2: 'car' ,3: 'motorcycle', 5: 'bus', 7: 'truck'

        #brand define
        load.brand= ['Audi', 'Chrysler', 'Citroen', 'GMC', 'Honda', 'Hyundai', 'Infiniti', 'Mazda', 'Mercedes', 'Mercury', 'Mitsubishi', 'Nissan', 'Renault', 'Toyota', 'Volkswagen', 'acura', 'bmw', 'cadillac', 'chevrolet', 'dodge', 'ford', 'jeep', 'kia', 'lexus', 'lincoln', 'mini', 'porsche', 'ram', 'range rover', 'skoda', 'subaru', 'suzuki', 'volvo']
        
        #colour classes
        load.colour_class = ['beige','black','blue','brown','gold','green','grey','orange','pink','purple','red','silver','tan','white','yellow']
        
    def open_folder_csv(load):
        #open new folder
        load.new_folder_path = f'save/{data_and_time()}'
        os.makedirs(load.new_folder_path)
        
        #create csv file
        load.csv_file_path = create_csv(load.new_folder_path)
    
    def run_detaction(load,table):
        
        #copy a new frame for plotting 
        new_frame = load.frame.copy()
                
        # using YOLOV8 to detact the vehicle and the License plate
        vehicle_detaction_results = load.vehicel_model(load.frame)[0]
            
        for vehicle_detection in vehicle_detaction_results.boxes.data.tolist():
            vx1, vy1, vx2, vy2, vscore, vclass_id = vehicle_detection
                        
            #get the correct classes and the the predict score more that equal to 80%
            if int(vclass_id) in load.vehicles and vscore >= 0.8:
                    
                #crop out the vehicle frame    
                vehicle_crop = load.frame[int(vy1):int(vy2), int(vx1): int(vx2), :]
                    
                #predict the vehicle plate area    
                car_plate_results = load.plate_detection(vehicle_crop)[0]                   
                for detection in car_plate_results.boxes.data.tolist():
                    px1,py1, px2, py2, pscore, classid = detection                       
                          
                    #run if the acuraccy predict is more that equal to % 
                    if pscore >= 0.8:
                            
                        #crop out the lp
                        lp_crop = vehicle_crop[int(py1):int(py2), int(px1): int(px2)]
                            
                        vehicle_plate = text_reconise(lp_crop,load.reader)
                            
                        if vehicle_plate[0]:
                                
                            #start brand detaction
                            v_brand  = "" 
                                
                            brand_detaction_results = load.brand_detection(vehicle_crop)[0]

                            for detection in brand_detaction_results.boxes.data.tolist():
                                bx1,by1, bx2, by2, bscore, bclassid = detection

                                if bscore >= 0.7:  
                                    v_brand = load.brand[int(bclassid)]
                                        
                            color_result = load.colour_class[color_reconigse(vehicle_crop,load.color_model,load.device)]
                                
                            result_data = [vehicle_plate[1],load.vehicles[vclass_id],v_brand,color_result]   
                                    
                            #check the vehicle plate duplicated
                            noduplicate,no = check_duplicated_plate_numbers_no(load.csv_file_path,vehicle_plate[1]) #return array 0 : true/false , 1: index number
                                
                            insert_table_info(table,result_data,f'{load.new_folder_path}/{vehicle_plate[1]}.jpg')
                                
                            #if no duplicated 
                            if noduplicate:
                                    
                                #check the illger vehicle and return warnning message if illger
                                invalid,message,vehicle_onwer = check_invalid_vehicle(result_data,"utils/database.csv")
                                    
                                if invalid:
                                    insert_table_info(table,result_data,f'{load.new_folder_path}/{vehicle_plate[1]}.jpg',invalid,message,vehicle_onwer)
                                    
                                        
                                #insert data to the csv file
                                insert_csv(no,result_data,load.csv_file_path,message)
                                
                                #save the image with the lp name.                                        
                                cv2.imwrite(f'{load.new_folder_path}/{vehicle_plate[1]}.jpg', vehicle_crop)   
                                 
                            QApplication.processEvents() 
                                        
                            #plot lp
                            #drawbox(new_frame,int(vx1+px1),int(vx1+px1+(px2-px1)),int(vy1+py1),int(vy1+py1+(py2-py1)),f"{str(vehicle_plate[1])}_{round(vehicle_plate[2],2)}",(0, 0, 255), 5)
                                
                            #plot car
                            #drawbox(new_frame,int(vx1),int(vx2),int(vy1),int(vy2),f'{vehicles[vclass_id]}_{vehicle_plate[1]}',(255, 0, 0), 5)      
                            
        return new_frame 
    
    def video_detaction(load,table):
        
        #get the file path and csv path
        load.open_folder_csv()
        
        start_time = time.time()
        
        #open video
        cap = cv2.VideoCapture(table.detact_input)

        # Get the video frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object 
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 25.0, (frame_width, frame_height))
        
            
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, load.frame = cap.read()

            if success:
                
                new_frame = load.run_detaction(table) #call the function and return the new_version_frame
                                                        
                out.write(new_frame)# add the frame to the video.
            
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        out.release() 
        
        end_time = time.time()

        # Calculate the elapsed time
        running_time = end_time - start_time

        print("\tProgram running time:", running_time/60, "minutes")
        
        table.runing_text.setText(f"End \n Time Taken : \n{round(running_time/60 , 2)} minutes")
        table.text_container.setStyleSheet("background-color:lightgreen;")
        QApplication.processEvents() 
    
    def image_detaction(load,table):
        
        start_time = time.time()
        #get the file path and csv path
        load.open_folder_csv()
        
        load.frame = cv2.imread(table.detact_input)
        load.run_detaction(table)

        end_time = time.time()

        # Calculate the elapsed time
        running_time = end_time - start_time

        print("\nProgram running time:", running_time/60, "minutes")
        
        table.runing_text.setText(f"End \n Time Taken : \n{round(running_time/60 , 2)} minutes")
        table.text_container.setStyleSheet("background-color:lightgreen;")
        QApplication.processEvents() 
    
    def live_detaction(load,table):
        
        #get the file path and csv path
        load.open_folder_csv()
        
        start_time = time.time()
        
        #open video
        cap = cv2.VideoCapture(table.input)

        # Get the video frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object 
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 25.0, (frame_width, frame_height))
        
            
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, load.frame = cap.read()

            if success:
                
                new_frame = load.run_detaction(table) #call the function and return the new_version_frame
                                                        
                out.write(new_frame)# add the frame to the video.
            
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        out.release() 
        
        
        end_time = time.time()

        # Calculate the elapsed time
        running_time = end_time - start_time

        print("\tProgram running time:", running_time/60, "minutes")
        
        table.runing_text.setText(f"End \n Time Taken : \n{round(running_time/60 , 2)} minutes")
        table.text_container.setStyleSheet("background-color:lightgreen;")
        QApplication.processEvents() 
        

