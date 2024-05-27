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
    QAbstractItemView
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

    data_with_index = [str(no)] + data + message
    
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

def insert_table_info(self,data,img_path,invalid=False,message="",vehicle_onwer=""):
    
    if invalid : choice = self.table_warnning 
    else : choice = self.table_info
    
    row_count = choice.rowCount()
    choice.insertRow(row_count)
    
    # Add image to the Photo column
    image_path = img_path
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
        

def run(self):
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(self.input)

    #open new folder
    new_folder_path = f'save/{data_and_time()}'
    os.makedirs(new_folder_path)
    
    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(f'{new_folder_path}/result_video.avi', fourcc, 25.0, (frame_width, frame_height))
    
    # Load the model
    vehicel_model = YOLO('utils/yolov8n.pt')
    plate_detection = YOLO('utils/car_plate_v2.pt')
    brand_detection = YOLO('utils/brand.pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   # Load the model for color recorigse
    color_model = models.googlenet(pretrained=False, aux_logits=True)  # Set aux_logits to True to match the saved model
    num_ftrs = color_model.fc.in_features
    color_model.fc = nn.Linear(num_ftrs, 15)  # Adjust num_classes to match your dataset
    color_model.aux1.fc2 = nn.Linear(color_model.aux1.fc2.in_features, 15)
    color_model.aux2.fc2 = nn.Linear(color_model.aux2.fc2.in_features, 15)

    model_path = "utils/colour.pth"
    color_model.load_state_dict(torch.load(model_path))
    color_model = color_model.to(device)
    color_model.eval()  # Set the model to evaluation mode

    
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'], gpu=True)
    
    vehicles = {2:"car",5:'bus', 7:'truck'} # 2: 'car' ,3: 'motorcycle', 5: 'bus', 7: 'truck'

    #brand define
    brand= ['Audi', 'Chrysler', 'Citroen', 'GMC', 'Honda', 'Hyundai', 'Infiniti', 'Mazda', 'Mercedes', 'Mercury', 'Mitsubishi', 'Nissan', 'Renault', 'Toyota', 'Volkswagen', 'acura', 'bmw', 'cadillac', 'chevrolet', 'dodge', 'ford', 'jeep', 'kia', 'lexus', 'lincoln', 'mini', 'porsche', 'ram', 'range rover', 'skoda', 'subaru', 'suzuki', 'volvo']
    
    #colour classes
    colour_class = ['beige','black','blue','brown','gold','green','grey','orange','pink','purple','red','silver','tan','white','yellow']
    
    #create csv file
    csv_file_path = create_csv(new_folder_path)
        
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            
            #copy a new frame for plotting 
            new_frame = frame.copy()
            
            # using YOLOV8 to detact the vehicle and the License plate

            # run the vehicle detatction
            vehicle_detaction_results = vehicel_model(frame)[0]
            
            for vehicle_detection in vehicle_detaction_results.boxes.data.tolist():
                vx1, vy1, vx2, vy2, vscore, vclass_id = vehicle_detection
                        
                #get the correct classes and the the predict score more that equal to 80%
                if int(vclass_id) in vehicles and vscore >= 0.8:
                    
                    #crop out the vehicle frame    
                    vehicle_crop = frame[int(vy1):int(vy2), int(vx1): int(vx2), :]
                    
                    #predict the vehicle plate area    
                    car_plate_results = plate_detection(vehicle_crop)[0]
                    
                    drawbox(new_frame,int(vx1),int(vx2),int(vy1),int(vy2),f'{round(vscore,2)}',(255, 0, 0), 5)                    
                    for detection in car_plate_results.boxes.data.tolist():
                        px1,py1, px2, py2, pscore, classid = detection                       
                        
                        drawbox(new_frame,int(vx1+px1),int(vx1+px1+(px2-px1)),int(vy1+py1),int(vy1+py1+(py2-py1)),f"_{round(pscore,2)}",(0, 0, 255), 5)    
                        #run if the acuraccy predict is more that equal to % 
                        if pscore >= 0.8:
                            
                            #crop out the lp
                            lp_crop = vehicle_crop[int(py1):int(py2), int(px1): int(px2)]
                            
                            vehicle_plate = text_reconise(lp_crop,reader)
                            
                            if vehicle_plate[0]:
                                
                                #start brand detaction
                                v_brand  = "" 
                                
                                brand_detaction_results = brand_detection(vehicle_crop)[0]

                                for detection in brand_detaction_results.boxes.data.tolist():
                                    bx1,by1, bx2, by2, bscore, bclassid = detection
                                    drawbox(new_frame,int(bx1),int(bx2),int(by1),int(by2),f'{brand[int(bclassid)]}_{bscore}',(0, 255, 0), 5)

                                    if bscore >= 0.7:  
                                        v_brand = brand[int(bclassid)]
                                        
                                color_result = colour_class[color_reconigse(new_frame,color_model,device)]
                                
                                result_data = [vehicle_plate[1],vehicles[vclass_id],v_brand,color_result]   
                                    
                                #check the vehicle plate duplicated
                                noduplicate,no = check_duplicated_plate_numbers_no(csv_file_path,vehicle_plate[1]) #return array 0 : true/false , 1: index number
                                
                                insert_table_info(self,result_data,f'{new_folder_path}/{vehicle_plate[1]}.jpg')
                                
                                #if no duplicated 
                                if(noduplicate):
                                    
                                    #check the illger vehicle and return warnning message if illger
                                    invalid,message,vehicle_onwer = check_invalid_vehicle(result_data,"utils/database.csv")
                                    
                                    if invalid:
                                        insert_table_info(self,result_data,f'{new_folder_path}/{vehicle_plate[1]}.jpg',invalid,message,vehicle_onwer)
                                        
                                    #insert data to the csv file
                                    insert_csv(no,result_data,csv_file_path,message)
                                
                                    #save the image with the lp name.                                        
                                    cv2.imwrite(f'{new_folder_path}/{vehicle_plate[1]}.jpg', vehicle_crop)   
                                
                                    
                                QApplication.processEvents()
                                
                                        
                                #plot lp
                                #drawbox(new_frame,int(vx1+px1),int(vx1+px1+(px2-px1)),int(vy1+py1),int(vy1+py1+(py2-py1)),f"{str(vehicle_plate[1])}_{round(vehicle_plate[2],2)}",(0, 0, 255), 5)
                                
                                #plot car
                                #drawbox(new_frame,int(vx1),int(vx2),int(vy1),int(vy2),f'{vehicles[vclass_id]}_{vehicle_plate[1]}',(255, 0, 0), 5)       
                                                       
            out.write(new_frame)
            #if show_output(new_frame): break             

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    
    end_time = time.time()

    # Calculate the elapsed time
    running_time = end_time - start_time

    print("\tProgram running time:", running_time/60, "minutes")
        
class result_window(QMainWindow):
    def __init__(self,input):
        super().__init__()

        self.setWindowTitle("Result")
        self.setMinimumSize(QSize(1000, 600)) 

        self.input = input
        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        self.label_result_table = QLabel("Detact Result")
        self.label_result_table.setAlignment(Qt.AlignCenter)
        self.label_result_table.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.label_result_table)
        
        self.table_info = QTableWidget()

        self.table_info.setColumnCount(5)
        self.table_info.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_info.setHorizontalHeaderLabels(['Vehicle_Image',"License_Plate","Vehicle_Type", "Vehicle_Brand",'Vehicle_Colour'])

        self.table_info.setAlternatingRowColors(True)
        
        self.table_info.setStyleSheet("""
            QTableWidget::item:!alternate {
                background-color: #f2f2f2; /* Light gray for odd rows */
            }
            QTableWidget::item:alternate {
                background-color: white; /* White for even rows */
            }
        """)
        
        layout.addWidget(self.table_info)
        
        self.label_table_warnning = QLabel("Warning")
        self.label_table_warnning.setAlignment(Qt.AlignCenter)
        self.label_table_warnning.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.label_table_warnning)
        
        
        self.table_warnning = QTableWidget()
        self.table_warnning.setColumnCount(5)
        self.table_warnning.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_warnning.setHorizontalHeaderLabels(['Vehicle_Image',"License_Plate","Vehicle_Type", "Vehicle_Brand",'Vehicle_Colour','Warning Message','Vehicle_Onwner'])
        self.table_warnning.setAlternatingRowColors(True)
        self.table_warnning.setStyleSheet("""
            QTableWidget::item:!alternate {
                background-color: #f2f2f2; /* Light gray for odd rows */
            }
            QTableWidget::item:alternate {
                background-color: white; /* White for even rows */
            }
        """)
        
        layout.addWidget(self.table_warnning)

        
    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, lambda:run(self))
