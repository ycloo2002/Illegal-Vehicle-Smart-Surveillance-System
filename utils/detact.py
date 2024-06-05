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
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

from PySide6.QtWidgets import (
    QApplication,
    QTableWidgetItem,
)
from PySide6.QtGui import QPixmap, QIcon,QImage
from PySide6.QtCore import QObject,Slot,Signal, QMutex, QMutexLocker
    
def display_img(display,image):
    """display the image in the result page

    Args:
        display (_type_): get classes that contain the gui argument
        image (_type_): the image with the cv2 format
    """
    #turn the cv2 format to the pixmap format
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    pixmap = QPixmap.fromImage(q_image)
    
    #set the image to the gui
    display.label_img.setPixmap(pixmap)

def drawbox(img,x1,x2,y1,y2,label_text,color = (255, 0, 0),thickness = 5):
    """draw the rectangle box to the image with label

    Args:
        img (_type_): image input
        x1 (int): x1 location
        x2 (int): x2 location
        y1 (int): y1 location
        y2 (int): y2 location
        label_text (str): the label
        color (tuple, optional): color of the line . Defaults to (255, 0, 0).
        thickness (int, optional): the thickness of line . Defaults to 5.
    """
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
    
    #draw the rectangle
    start_point = label_position
    end_point = (label_position[0] + label_width, label_position[1] - label_height)
    cv2.rectangle(img, start_point, end_point, color, thickness=cv2.FILLED)

    #putting the label text
    cv2.putText(img, label_text, label_position, font, font_scale, font_color, thickness=5)

def data_and_time():
    """get the date and time as the '%Y-%m-%d %H-%M-%S' format

    Returns:
        str: Type of the date and time '%Y-%m-%d %H-%M-%S' 
    """
    date = datetime.now()
    return str(date.strftime('%Y-%m-%d %H-%M-%S')) 
    
def insert_csv(no,data,csv_path,message):
    """insert the text to the csv file

    Args:
        no (int): the no 
        data (array): the array should contain plate number, vehicle type,vehicle model and vehicle colour
        csv_path (str): the input csv file path
        message (str): the warnning message that insert at the last column
    """
    data_with_index = [str(no)] + data + [message]
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_with_index)   
    
def create_csv(path):
    """create the csv file for saving the detection result

    Args:
        path (str): the folder path

    Returns:
        str: the csv path that created
    """
    csv_file_path = f"{path}/result.csv"
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['No','Plate_number', 'Type','Brand','Colour','Message'])
        print(f"CSV file '{csv_file_path}' successful create.")
        
    return csv_file_path
            
def text_reader(reader,img):
    """Function to convert the LP image to text.

    Args:
        reader (_type_): model for the reader
        img (_type_): image for the detection

    Returns:
        bool: the result 
        str: LP text
    """
    #define the pattern
    pattern =r'^[a-zA-Z][a-zA-Z0-9]*[0-9][a-zA-Z0-9]*$'
    
    #read the text
    text_result =reader.readtext(img)              
    #if non of the result showing out  
      
    if text_result:          
        total_score =0
        lp_text = "" 
               
        #combine the text
        for (bbox, text, score) in text_result:
            lp_text += text
            total_score += score
                                        
        avg_lp_score = total_score/len(text_result)
        
        lp_text = lp_text.upper().replace(' ', '')
    
        #continue when the pattern is correct.
        if re.match(pattern, lp_text) and avg_lp_score > 0.5:
            return True,lp_text
    
    return False,""    
                    
def search_vehicle(frame,vehicel_model,plate):
    """Function to detect the vehicle with the YOLO.Return all the result with 0.8 accuracy

    Args:
        frame (_type_): image
        vehicel_model (_type_): YOLO model for the vehicle detection
        plate(array): the array contain the plate information
    Returns:
        array: x1,y1,x2,y2,score,classid
    """
    vehicle_detect = []
    car_results = vehicel_model(frame,classes=[2,5,7])[0]                 
    for detection in car_results.boxes.data.tolist():
        cx1,cy1, cx2, cy2, pscore, classid = detection 
          
        if plate[0] > cx1 and plate[1] > cy1 and plate[2] < cx2 and plate[3] < cy2:
            vehicle_detect = [cx1,cy1, cx2, cy2, classid]
            break

    return vehicle_detect

def search_plate(frame,plate_detection,reader,save_plate):
    """detect the LP and call the function to read LP text. 

    Args:
        frame (_type_): image
        plate_detection (_type_): LP YOLO model
        reader (_type_): Reader model
        save_plate (array): the array that saving the plate_no

    Returns:
        array: the location of x1,y1,x2,y2 and the result
    """
    plate_detect = []
    
    car_plate_results = plate_detection(frame)[0]                   
    for detection in car_plate_results.boxes.data.tolist():
        px1,py1, px2, py2, pscore, classid = detection                       
                          
        if pscore >= 0.8:
                        
            lp_crop = frame[int(py1):int(py2), int(px1): int(px2)]    

            reader_result,result = text_reader(reader,lp_crop)
            
            if reader_result:
                
                same_lp = False
                for plate_no in save_plate:
                    if result == plate_no:
                        same_lp = True
                        break
                
                if not same_lp : plate_detect.append([px1,py1,px2,py2,result])
    
    return plate_detect

def search_brand(brand_detection,img):
    """Function to detect the brand

    Args:
        brand_detection (_type_): Yolo model for the brand detection
        img (_type_): image

    Returns:
        int: brand id
    """
    brand_detaction_results = brand_detection(img)[0]

    for detection in brand_detaction_results.boxes.data.tolist():
        bx1,by1, bx2, by2, bscore, bclassid = detection

        return int(bclassid)
        
def color_reconigse(image, model,device):
    """Vehicle colour recognition

    Args:
        image (_type_): img input cv2 format
        model (_type_): the cnn - googlenet model (colour)
        device (_type_): device gpu info

    Returns:
        int: result
    """
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
    """Insert data to gui

    Args:
        table (_type_): class of gui
        data (array): value to insert
        image_path (str): img path
        invalid (bool, optional): write in illegal table or not. Defaults to False.
        message (str, optional): illegal messahe. Defaults to "".
        vehicle_onwer (str, optional): onwer name. Defaults to "".
    """
    if invalid : choice = table.table_warnning 
    else : choice = table.table_info
    
    row_count = choice.rowCount()
    choice.insertRow(row_count)
    
    #insert image
    item = QTableWidgetItem()
    pixmap = QPixmap(image_path).scaled(200, 200)  # Resize the image
    icon = QIcon(pixmap)
    item.setIcon(icon)

    choice.setIconSize(pixmap.size())

    # Set a fixed size hint for the item to ensure it is displayed properly
    item.setSizeHint(pixmap.size())
        
    choice.setItem(row_count, 0, item)
        
    # Optionally set row height and column width to ensure the image fits
    choice.setRowHeight(row_count, 210)
    choice.setColumnWidth(0, 210)
    
    choice.setItem(row_count, 1, QTableWidgetItem(data[0]))
    choice.setItem(row_count, 2, QTableWidgetItem(data[1]))
    choice.setItem(row_count, 3, QTableWidgetItem(data[2]))
    choice.setItem(row_count, 4, QTableWidgetItem(data[3]))
    
    if invalid:
        choice.setItem(row_count, 5, QTableWidgetItem(message))
        choice.setItem(row_count, 6, QTableWidgetItem(vehicle_onwer))
                                
def check_invalid_vehicle(result_data,path):
    """check the illegal vehicle and return any warnning message

    Args:
        result_data (array): value
        path (str): the database csv path

    Returns:
        bool: any illegal vehicle
        str: illegal message
        str: the vehicle onwer name
    """
    #default
    code = "notfound"
    onwer_name = ""
    a_typr = ""
    a_brand = ""
    a_color = ""
    #open the database file
    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for col in reader:
            
            if col[2] == result_data[0]: # true in vehicle plate 
                code = "r"
                onwer_name = col[1]
                #type
                if col[3] != result_data[1]: 
                    code += "T"
                    a_typr = col[3]
                
                #brand
                if col[4] != result_data[2]: 
                    code += "B"
                    a_brand = col[4]
                
                #color
                if col[5] != result_data[3]: 
                    code += "C"
                    a_color = col[5]
                
                break
        
        match code:
            case "notfound" : return True,"This License Plate not register in the system.",onwer_name
            
            case "rTBC" : return True,f"Invalid Vehicle Type, Brand and Colour for this License Plate.\nThe register type is '{a_typr}' ,brand is '{a_brand}' and colour is '{a_color}'",onwer_name
            case "rTB" : return True,f"Invalid Vehicle Type, Brand and Colour for this License Plate. \nThe register type is '{a_typr}' and brand is '{a_brand}'",onwer_name
            case "rTC" : return True,f"Invalid Vehicle Type and Colour for this License Plate. \nThe register type is '{a_typr}' and colour is '{a_color}'",onwer_name
            case "rT" : return True,f"Invalid Vehicle Type for this License Plate. \nThe register colour is '{a_typr}'",onwer_name
                
            case "rB" : return True,f"Invalid Vehicle Brand for this License Plate. \nThe register brand is '{a_brand}'",onwer_name
            case "rBC" : return True,f"Invalid Vehicle Brand and Colour for this License Plate. \nThe register brand is '{a_brand}' and colour is '{a_color}'",onwer_name
                
            case "rC" : return True,f"Invalid Vehicle Colour for this License Plate. \nThe register colour is '{a_color}'",onwer_name
            
            case "r" : return False,"No error found.",onwer_name

class Load_Object():
    """Load the nessasry item
    """
    def __init__(load):
        super().__init__()
        utils_basedir = os.path.dirname(__file__)
        
        print("\nload the nessary item")
        
        # Load the model
        load.vehicel_model = YOLO(f'{utils_basedir}/model/yolov8n.pt')
        print("\nSuccessfully load vehicle model")
        
        load.plate_detection = YOLO(f'{utils_basedir}/model/car_plate_v4.pt')
        print("\nSuccessfully load plate model")
        
        load.brand_detection = YOLO(f'{utils_basedir}/model/brand_v3.pt')
        print("\nSuccessfully load brand model")
        
    
        # Load the model for color recorigse
        color_model = models.googlenet(pretrained=False, aux_logits=True)  # Set aux_logits to True to match the saved model
        num_ftrs = color_model.fc.in_features
        color_model.fc = nn.Linear(num_ftrs, 15)  # Adjust num_classes to match your dataset
        color_model.aux1.fc2 = nn.Linear(color_model.aux1.fc2.in_features, 15)
        color_model.aux2.fc2 = nn.Linear(color_model.aux2.fc2.in_features, 15)

        model_path = f'{utils_basedir}/model/colour.pth'
        color_model.load_state_dict(torch.load(model_path))
        load.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        color_model = color_model.to(load.device)
        color_model.eval()  # Set the model to evaluation mode
        
        load.color_model = color_model
        print("\nSuccessfully load colour model")
        
        # Initialize the OCR reader
        load.reader = easyocr.Reader(['en'], gpu=True)
        
        print("\nSuccessfully load reader model")
        
        
        load.vehicles = {2:"car",5:'bus', 7:'truck'} # 2: 'car' ,3: 'motorcycle', 5: 'bus', 7: 'truck'
        print("\n Successfully load vehicle : ",load.vehicles)
        
        #brand define
        load.brand= ['Audi', 'Chrysler', 'Citroen', 'GMC', 'Honda', 'Hyundai', 'Infiniti', 'Mazda', 'Mercedes', 'Mercury', 'Mitsubishi', 'Nissan', 'Renault', 'Toyota', 'Volkswagen', 'acura', 'bmw', 'cadillac', 'chevrolet', 'dodge', 'ford', 'jeep', 'kia', 'lexus', 'lincoln', 'mini', 'porsche', 'ram', 'range rover', 'skoda', 'subaru', 'suzuki', 'volvo','Proton','Perodua']
        print("\nSuccessfully load brand class : ",load.brand)
        
        #colour classes
        load.colour_class = ['beige','black','blue','brown','gold','green','grey','orange','pink','purple','red','silver','tan','white','yellow']
        print("\nSuccessfully load colour class : ",load.colour_class)
        
        load.database_path = f'{utils_basedir}/database.csv'
        print("\nSuccessfully load Database path : ",load.database_path)
        
        
        if not os.path.exists("save"):
            os.makedirs("save")
            print("Folder save created.")
    
class Detection(QObject):
    """ The main vehicle detection classes.
        Include the running detection with live,video and image.
    """
    warnning = Signal(str)
    finish = Signal(str)
    
    def __init__(load,define_object,source_path):
        super().__init__()

        for attr, value in define_object.load_object.__dict__.items():
            setattr(load, attr, value)
        load.gui = define_object
        load.detact_input = source_path
        
        load._is_running = True
        load.mutex = QMutex()

    def stop(load):
        """stop the detection

        Args:
            load (_type_): _description_
        """
        QMutexLocker(load.mutex)
        load._is_running = False
        
    def open_folder_csv(load):
        """Generate the folder and csv file

        Args:
            load (_type_): _description_
        """
        load.folder_name = data_and_time()
        #open new folder
        load.new_folder_path = f'result/{load.folder_name}'
        os.makedirs(load.new_folder_path)
        
        #open folder for save the crop image
        os.makedirs(f'{load.new_folder_path}/crop')
        
        #create csv file
        load.csv_file_path = create_csv(load.new_folder_path)
    
    def vehicle_illegal_detection(load):
        """run for the vehicle detection which include the vehicle type,brand and color
            
        Args:
            load (_type_): _description_

        Returns:
            _type_: new image
        """
        
        new_frame = load.frame.copy()
            
        plate_detect = search_plate(load.frame,load.plate_detection,load.reader,load.save_plate)
        
        if len(plate_detect) > 0:
            
            for plate in plate_detect:
                
                load.save_plate.append(plate[4])
                
                vehicle_detect,new_frame = search_vehicle(load.frame,load.vehicel_model,plate)

                if not vehicle_detect == "":
                    x1 = vehicle_detect[0]
                    y1 = vehicle_detect[1]
                    
                    x2 = vehicle_detect[2]
                    y2 = vehicle_detect[3]
                    
                    #crop the vehicle        
                    vehicle_crop = load.frame[int(y1):int(y2), int(x1): int(x2)]
                    
                    #detect the brand
                    detect_brand = load.brand[search_brand(load.brand_detection,vehicle_crop)]  
                    
                    #detect vehicle colour           
                    detect_colour = load.colour_class[color_reconigse(vehicle_crop,load.color_model,load.device)]
                    
                    result_data = [plate[4],load.vehicles[int(vehicle_detect[4])],detect_brand,detect_colour]  
                    
                    load.total_vehicle += 1 
                                
                    #save crop image path
                    img_path = f'{load.new_folder_path}/crop/{plate[4]}.jpg'
                                
                    #save the image with the lp name.                                        
                    cv2.imwrite(img_path, vehicle_crop)  
                                
                    insert_table_info(load.gui,result_data,img_path)
                                    
                    #check the illger vehicle and return warnning message if illger
                    invalid,message,vehicle_onwer = check_invalid_vehicle(result_data,load.database_path)
                                    
                    if invalid:
                        insert_table_info(load.gui,result_data,img_path,invalid,message,vehicle_onwer)
                        load.total_warnning +=1
                                    
                        #insert data to the csv file
                        insert_csv(len(load.save_plate),result_data,load.csv_file_path,message)
                                
                    QApplication.processEvents() 
                    #drawbox(new_frame,int(vehicle[0]),int(vehicle[2]),int(vehicle[1]),int(vehicle[3]),f'{plate[4]}',(255, 0, 0), 5) 
                                              
                                
        load.gui.runing_text.setText(f"Loading. \n Total {load.total_vehicle} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle.")    
        return new_frame 
                                               
    @Slot()
    def video_detaction(load):
        """This is the video detection function.It will get the input and covert the video to frame. Each frame will be call the run_detection to get the detection.
            Beside, it also will save the input as the new video after making some plotting.
        Args:
            load (_type_): _description_
        """
        load.total_vehicle = 0  
        
        load.total_warnning = 0
        
        #get the file path and csv path
        load.open_folder_csv()
        
        start_time = time.time()
        
        #open video
        cap = cv2.VideoCapture(load.detact_input)

        # Get the video frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object 
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 25.0, (frame_width, frame_height))
        
        load.save_plate = []    
        # Loop through the video frames
        while cap.isOpened():
            
            QMutexLocker(load.mutex)  # Ensure thread-safe access to _is_running
            if not load._is_running:
                print("Worker stopped.")
                break
            
            # Read a frame from the video
            success, load.frame = cap.read()

            if success:
                
                new_frame = load.vehicle_illegal_detection() #call the function and return the new_version_frame
                                                     
                out.write(new_frame)# add the frame to the video.
                
            else:
                # Break the loop if the end of the video is reached
                break
        
        # Release the video capture object and close the display window
        cap.release()
        out.release() 
        
        
        # Calculate the elapsed time
        running_time =time.time()- start_time

        print(f"Total {load.total_vehicle} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
        print("\nProgram running time:", running_time/60, "minutes")
        
        load.gui.runing_text.setText(f"detection End. \n Total {load.total_vehicle} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
        load.gui.text_container.setStyleSheet("background-color:lightgreen;")
        load.gui.result_home_btn.setEnabled(True)
        load.gui.stop_running_btn.setEnabled(False)
        load.finish.emit(load.folder_name)

    @Slot()
    def image_detaction(load):
        """This is the image detection function.It will get the input and call the run_detection to get the detection.
            Beside, it also will save the input.
        Args:
            load (_type_): _description_
        """
        load.total_vehicle = 0  
        
        load.total_warnning = 0
        
        start_time = time.time()
        #get the file path and csv path
        load.open_folder_csv()
        
        load.frame = cv2.imread(load.detact_input)
        cv2.imwrite(f'{load.new_folder_path}/original.png', load.frame)
        
        load.vehicle_illegal_detection()
        
        end_time = time.time()

        # Calculate the elapsed time
        running_time = end_time - start_time
        
        print(f"Total {load.total_vehicle} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
        print("\nProgram running time:", running_time/60, "minutes")
        
        load.gui.runing_text.setText(f"detection End. \n Total {load.total_vehicle} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
        load.gui.text_container.setStyleSheet("background-color:lightgreen;")
        load.gui.result_home_btn.setEnabled(True)
        load.gui.stop_running_btn.setEnabled(False)
        load.finish.emit(load.folder_name)
    
    @Slot()
    def live_detaction(load):
        """This is the live detection function.It will get the input and covert the video to frame. Each frame will be call the run_detection to get the detection.
            Beside, it also will save the input as the new video after end the live
        Args:
            load (_type_): _description_
        """
        #open video
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            load.warnning.emit("Unable to open the camera")
            
        else:
            load.total_vehicle = 0  
        
            load.total_warnning = 0
            
            start_time = time.time()
            
            #get the file path and csv path
            load.open_folder_csv()
            
            
            # Get the video frame width and height
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create VideoWriter object 
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 25.0, (frame_width, frame_height))
        
            while True:
                QMutexLocker(load.mutex)  # Ensure thread-safe access to _is_running
                if not load._is_running:
                    print("Worker stopped.")
                    break
            
                # Capture frame-by-frame
                ret, load.frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break
                       
                new_frame = load.vehicle_illegal_detection() #call the function and return the new_version_frame
                                                                            
                out.write(new_frame)# add the frame to the video.
                
                # Press 'q' on the keyboard to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Release the video capture object and close the display window
            cap.release()
            out.release() 
            
            end_time = time.time()

            # Calculate the elapsed time
            running_time = end_time - start_time

            print(f"Total {load.total_vehicle} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
            print("\nProgram running time:", running_time/60, "minutes")
            
            load.gui.runing_text.setText(f"detection End. \n Total {load.total_vehicle} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
            load.gui.text_container.setStyleSheet("background-color:lightgreen;")
            load.gui.result_home_btn.setEnabled(True)
            load.gui.stop_running_btn.setEnabled(False)
            load.finish.emit(load.folder_name)
        
